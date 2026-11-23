"""
entropy_reg2.py — v2：更正确的可微码率估计 + 多 seed + 多正则对照。

v1 的教训：
  - L1 在帕累托上打败了我 v1 的"直方图熵"估计器
  - 原因：直方图熵对每个权重的梯度是退化的（只有统计量在动），
    而真正要优化的是"每个权重自己的期望码长"。

v2 换成交叉熵形式：
    R = E_i[ -log2 P(q(w_i)) ]
  其中 P 用 detached 的软直方图（相当于码本分布），
  q(w_i) 用软指派（可微），于是梯度直接作用在每个权重上。

这和"量化 + 算术编码"的真实成本是同一个量。
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


def corpus(nbytes=8 << 20):
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
        j = ix[i].tolist()
        x = torch.stack([torch.from_numpy(data[k:k + ctx].astype(np.int64)) for k in j])
        y = torch.stack([torch.from_numpy(data[k + 1:k + 1 + ctx].astype(np.int64)) for k in j])
        yield x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        z = self.ln1(x)
        a, _ = self.att(z, z, z, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab=256, d=160, L=4, h=4, ctx=128):
        super().__init__()
        self.te = nn.Embedding(vocab, d)
        self.pe = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(L)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.ctx = ctx

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.te(x) + self.pe(pos)[None]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), 1)
        for b in self.blocks:
            h = b(h, mask)
        return self.head(self.ln(h))


# ---------------------------------------------------------------- 可微码率
def rate_loss(params, levels=16, span=3.0, tau=0.05):
    """E[-log2 p(q(w))]，梯度作用在每个权重上。"""
    tot = 0.0
    for p in params:
        if p.dim() < 2:
            continue
        w = p.reshape(-1)
        s = w.std().detach().clamp(min=1e-6)
        wn = w / s
        c = torch.linspace(-span, span, levels, device=w.device, dtype=w.dtype)
        logits = -((wn[:, None] - c[None, :]) ** 2) / tau
        q = torch.softmax(logits, dim=1)
        pr = q.mean(0).detach().clamp(min=1e-9)
        bits = -(q * torch.log2(pr)).sum(1)
        tot = tot + bits.mean()
    return tot


def hist_entropy(params, levels=16, span=3.0, tau=0.05):
    tot = 0.0
    for p in params:
        if p.dim() < 2:
            continue
        w = p.reshape(-1)
        s = w.std().detach().clamp(min=1e-6)
        wn = w / s
        c = torch.linspace(-span, span, levels, device=w.device, dtype=w.dtype)
        logits = -((wn[:, None] - c[None, :]) ** 2) / tau
        pr = torch.softmax(logits, dim=1).mean(0)
        tot = tot + (-(pr * torch.log2(pr + 1e-9)).sum())
    return tot


@torch.no_grad()
def measure_bits(model, levels=16):
    tot, n = 0.0, 0
    for p in model.parameters():
        w = p.detach().reshape(-1)
        a = w.abs().max().item()
        if a < 1e-8:
            n += w.numel()
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
def zero_frac(model):
    z = n = 0
    for p in model.parameters():
        if p.dim() < 2:
            continue
        a = p.abs().max().item()
        if a < 1e-8:
            continue
        step = a / 8.0
        q = torch.round(p / step)
        z += int((q == 0).sum())
        n += p.numel()
    return z / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--d", type=int, default=160)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--mode", default="wd",
                    choices=["wd", "l1", "hist", "rate", "l1rate"])
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw = np.frombuffer(corpus(), dtype=np.uint8)
    split = int(len(raw) * 0.95)
    tr, va = raw[:split], raw[split:]
    vs = list(batches(va, args.ctx, 24, 24, seed=99))

    torch.manual_seed(args.seed)
    model = GPT(d=args.d, L=args.layers, ctx=args.ctx).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)

    t0 = time.time()
    for i, (x, y) in enumerate(batches(tr, args.ctx, args.bs, args.steps, seed=args.seed + 1)):
        logits = model(x)
        ce = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        loss = ce
        if args.mode in ("l1", "l1rate"):
            reg = sum(p.abs().sum() for p in model.parameters() if p.dim() >= 2)
            loss = loss + args.lam * reg
        if args.mode == "hist" and args.lam > 0:
            loss = loss + args.lam * hist_entropy(model.parameters())
        if args.mode in ("rate", "l1rate") and args.lam > 0:
            loss = loss + args.lam * rate_loss(model.parameters())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        vl = 0.0
        for x, y in vs:
            vl += F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1)).item()
        vl /= len(vs)
    b16 = measure_bits(model, 16)
    b8 = measure_bits(model, 8)
    b32 = measure_bits(model, 32)
    zf = zero_frac(model)
    print(f"R,{args.mode},{args.lam},{args.seed},{vl/math.log(2):.5f},"
          f"{b16:.5f},{b8:.5f},{b32:.5f},{zf:.4f},{time.time()-t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
