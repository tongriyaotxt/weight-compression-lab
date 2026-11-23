"""
entropy_reg.py — 把"熵"写进损失函数，看能不能训练出可压缩的模型。

三组对照
  A  weight decay        现状
  B  L1                  标准的稀疏化手段
  C  熵正则               L = CE + λ·Σ H(q(W))

为什么 C 和 A/B 本质不同：
  熵(q(W)) 是**尺度不变**的 —— W 乘个常数，量化后码流分布不变。
  所以熵正则没法靠"把权重缩小"来作弊，它只能改变**分布的形状**。
  weight decay 和 L1 都能被"缩小"部分满足，熵不能。

评价指标不是 loss，而是 (val bits/char, bits/weight) 的帕累托前沿。
"""

import argparse
import glob
import math
import os
import sys
import sysconfig
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- data
def corpus(nbytes=6 << 20):
    root = sysconfig.get_paths()["stdlib"]
    buf = bytearray()
    for p in sorted(glob.glob(os.path.join(root, "*.py"))):
        try:
            buf += open(p, "rb").read()
        except OSError:
            continue
        if len(buf) >= nbytes:
            break
    return bytes(buf[:nbytes])


def batches(data, ctx, bs, steps, seed=0):
    g = torch.Generator().manual_seed(seed)
    ix = torch.randint(0, len(data) - ctx - 1, (steps, bs), generator=g)
    for i in range(steps):
        idx = ix[i]
        x = torch.stack([torch.from_numpy(data[j:j + ctx].astype(np.int64)) for j in idx.tolist()])
        y = torch.stack([torch.from_numpy(data[j + 1:j + 1 + ctx].astype(np.int64)) for j in idx.tolist()])
        yield x.to(DEV), y.to(DEV)


# ---------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        h = self.ln1(x)
        a, _ = self.att(h, h, h, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab=256, d=128, L=3, h=4, ctx=96):
        super().__init__()
        self.ctx = ctx
        self.te = nn.Embedding(vocab, d)
        self.pe = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(L)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.te(x) + self.pe(pos)[None]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), 1)
        for b in self.blocks:
            z = b.ln1(h)
            a, _ = b.att(z, z, z, attn_mask=mask, need_weights=False)
            h = h + a
            h = h + b.mlp(b.ln2(h))
        return self.head(self.ln(h))


# ---------------------------------------------------------------- entropy
def soft_entropy_group(params, levels=16, span=3.0, tau=0.05):
    """可微的经验熵：soft histogram over 归一化后的权重。"""
    tot = 0.0
    for p in params:
        if p.dim() < 2:
            continue
        w = p.reshape(-1)
        s = w.std().detach().clamp(min=1e-6)
        wn = w / s
        c = torch.linspace(-span, span, levels, device=w.device)
        d = (wn[:, None] - c[None, :]) ** 2
        pr = torch.softmax(-d / tau, dim=1).mean(0)
        tot = tot + (-(pr * torch.log(pr + 1e-9)).sum())
    return tot


@torch.no_grad()
def measure_bits(model, levels=16):
    tot, n = 0.0, 0
    for p in model.parameters():
        if p.dim() < 2:
            continue
        w = p.detach().reshape(-1)
        a = w.abs().max().item()
        if a < 1e-8:
            continue
        step = a / (levels / 2)
        q = torch.clamp(torch.round(w / step), -levels // 2, levels // 2 - 1).long()
        cnt = torch.bincount(q + levels // 2, minlength=levels).float()
        pr = cnt / cnt.sum()
        pr = pr[pr > 0]
        H = float(-(pr * torch.log2(pr)).sum())
        tot += H * w.numel()
        n += w.numel()
    return tot / max(n, 1)


@torch.no_grad()
def sparsity(model):
    z, n = 0, 0
    for p in model.parameters():
        if p.dim() < 2:
            continue
        z += int((p.abs() < 1e-6).sum())
        n += p.numel()
    return z / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=96)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--mode", default="wd", choices=["wd", "l1", "ent"])
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    raw = np.frombuffer(corpus(), dtype=np.uint8)
    split = int(len(raw) * 0.9)
    tr, va = raw[:split], raw[split:]

    torch.manual_seed(0)
    model = GPT(d=args.d, L=args.layers, ctx=args.ctx).to(DEV)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    t0 = time.time()
    vs = list(batches(va, args.ctx, 16, 20, seed=99))
    for i, (x, y) in enumerate(batches(tr, args.ctx, args.bs, args.steps, seed=1)):
        logits = model(x)
        ce = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        loss = ce
        if args.mode == "ent" and args.lam > 0:
            loss = loss + args.lam * soft_entropy_group(model.parameters())
        elif args.mode == "l1" and args.lam > 0:
            reg = sum(p.abs().sum() for p in model.parameters() if p.dim() >= 2)
            loss = loss + args.lam * reg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (i + 1) % 200 == 0:
            print(f"    step {i+1}/{args.steps}  ce={ce.item():.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        vl = 0.0
        for x, y in vs:
            vl += F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1)).item()
        vl /= len(vs)
    b16 = measure_bits(model, 16)
    b8 = measure_bits(model, 8)
    b32 = measure_bits(model, 32)
    sp = sparsity(model)
    dt = time.time() - t0
    print(f"RESULT tag={args.tag or args.mode} mode={args.mode} lam={args.lam} "
          f"params={nparam} val_bpc={vl/math.log(2):.4f} "
          f"bits16={b16:.4f} bits8={b8:.4f} bits32={b32:.4f} "
          f"sparsity={sp*100:.2f}% time={dt:.1f}s dev={DEV}", flush=True)


if __name__ == "__main__":
    main()
