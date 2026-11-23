"""
dz_train.py — dead-zone quantizer + rate objective, quantizer in the loop (QAT).

Question left open last round:
  L1 beat the soft-histogram rate regularizer because its gradient (+-1) pushes
  weights to EXACT zero, while the rate surrogate has zero gradient at the level
  centre. So: put the dead zone INSIDE the quantizer (centre bin width = dz * step),
  make dz and step learnable, objective = task loss + lam * H.
  Does it beat L1?

Design:
  1. Single-variable control: every QAT run has a learnable per-tensor step (LSQ).
     The only difference is whether dz is learnable. Otherwise we cannot tell
     "credit for the dead zone" from "credit for learning the scale".
  2. Deploy consistency: the quantizer used in training is the one used for
     evaluation; the code stream it produces is what gets entropy coded.
     So `bits` is the real code-stream entropy, not an MSE proxy.
  3. dz = 1 + softplus(rho) >= 1  (dz = 1 is standard mid-tread INT4).

All Linear weights are concatenated into one flat vector and step/dz are
expanded through a precomputed index, so the quantizer costs ~10 kernels
instead of 19 tensors x 6 kernels.

Output: labs/dz/runs.csv (one row per run), weights in labs/dz/models/
"""

import argparse
import glob
import math
import os
import sysconfig
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
K = 7  # 7 levels per side -> 15 codewords (symmetric, 0 is its own bin)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dz")
os.makedirs(os.path.join(OUT, "models"), exist_ok=True)


# ------------------------------------------------------------------ data
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


# ------------------------------------------------------------------ flat quantizer
class FlatQ(nn.Module):
    """One (log_step, rho) pair per Linear, applied on the flattened concatenation."""

    def __init__(self, mods):
        super().__init__()
        self.mods = mods
        self.shapes = [m.weight.shape for m in mods]
        self.sizes = [m.weight.numel() for m in mods]
        n = sum(self.sizes)
        el = torch.cat([torch.full((s,), i, dtype=torch.long) for i, s in enumerate(self.sizes)])
        self.register_buffer("el", el, persistent=False)
        self.register_buffer(
            "starts",
            torch.tensor(np.cumsum([0] + self.sizes[:-1]).astype(np.int64)),
            persistent=False,
        )
        init = torch.tensor(
            [math.log(max(m.weight.detach().abs().max().item() / K, 1e-6)) for m in mods]
        )
        self.log_step = nn.Parameter(init.to(DEV))
        self.rho = nn.Parameter(torch.full((len(mods),), -3.0, device=DEV))  # dz ~ 1.05
        self.register_buffer("gain", torch.tensor(0.0, device=DEV))
        self.n = n

    def dz(self):
        return 1.0 + F.softplus(self.rho)

    def step(self):
        return torch.exp(self.log_step + self.gain)


def dzq(w, step, dz, hard=True, clip_ste=False):
    """Elementwise dead-zone quantization with STE. w/step/dz broadcast.

    clip_ste=True: 饱和区（|w| 超过最高电平）的梯度置零 —— 经典 QAT 修补，
    用来诊断"权重逃逸到轨道"是不是纯训练伪影。
    """
    t = w.abs() / step
    k = torch.clamp(torch.floor(t - dz / 2.0) + 1.0, 0.0, float(K))
    mag = torch.where(k > 0, (dz / 2.0 + k - 0.5) * step, torch.zeros_like(t))
    q = torch.sign(w) * mag
    if not hard:
        return q
    out = w + (q - w).detach()
    if clip_ste:
        sat = (t > float(K)).detach()
        out = torch.where(sat, q.detach(), out)
    return out


def soft_rate(u, dz, tau=0.05):
    """u = w/step already flattened, dz elementwise -> expected code length per weight."""
    k = torch.arange(-K, K + 1, device=u.device, dtype=u.dtype)
    r = torch.sign(k) * torch.clamp(k.abs() - 0.5 + dz[:, None] / 2.0, min=0.0)
    logits = -((u[:, None] - r) ** 2) / tau
    q = torch.softmax(logits, dim=1)
    p = q.mean(0).detach().clamp(min=1e-9)
    return -(q * torch.log2(p)).sum(1)


def catw(qd):
    return torch.cat([m.weight.reshape(-1) for m in qd.mods])


def quant_all(qd, dz_on=True, clip_ste=False):
    flat = catw(qd)
    step = qd.step()[qd.el]
    dz = qd.dz()[qd.el] if dz_on else torch.ones_like(step)
    qflat = dzq(flat, step, dz, clip_ste=clip_ste)
    out = {}
    for i, m in enumerate(qd.mods):
        s = qd.starts[i].item()
        n = qd.sizes[i]
        out[m] = qflat[s:s + n].view(qd.shapes[i])
    return out


def huff_avg(p):
    import heapq

    h = [(float(x), i, {i: 0}) for i, x in enumerate(p) if x > 0]
    if len(h) <= 1:
        return 1.0
    heapq.heapify(h)
    depth = {i: 0 for i, x in enumerate(p) if x > 0}
    while len(h) > 1:
        a = heapq.heappop(h)
        b = heapq.heappop(h)
        for s in a[2]:
            depth[s] += 1
        for s in b[2]:
            depth[s] += 1
        heapq.heappush(h, (a[0] + b[0], a[1] + b[1], {**a[2], **b[2]}))
    return float(sum(p[i] * depth[i] for i in depth))


@torch.no_grad()
def measure_fast(qd, dz_on=True, per_tensor=16384):
    """抽样版码流熵，给 CBR 控制器用（每 20 步一次，必须便宜）。"""
    flat = catw(qd).detach()
    step = qd.step()[qd.el]
    dz = qd.dz()[qd.el] if dz_on else torch.ones_like(step)
    t = flat.abs() / step
    k = torch.clamp(torch.floor(t - dz / 2.0) + 1.0, 0.0, float(K))
    k = (torch.where(flat < 0, -k, k) + K).long()
    tot, n = 0.0, 0
    for i in range(len(qd.mods)):
        s, m = qd.starts[i].item(), qd.sizes[i]
        seg = k[s:s + m]
        if m > per_tensor:
            seg = seg[torch.randint(0, m, (per_tensor,), device=seg.device)]
        cnt = torch.bincount(seg, minlength=2 * K + 1).float()
        p = (cnt / cnt.sum()).cpu().numpy()
        nz = p[p > 0]
        tot += float(-(nz * np.log2(nz)).sum()) * m
        n += m
    return tot / n


@torch.no_grad()
def measure(qd, dz_on=True):
    """Per-tensor entropy + per-tensor Huffman length, size-weighted."""
    flat = catw(qd).detach()
    step = qd.step()[qd.el]
    dz = qd.dz()[qd.el] if dz_on else torch.ones_like(step)
    t = flat.abs() / step
    k = torch.clamp(torch.floor(t - dz / 2.0) + 1.0, 0.0, float(K))
    k = (torch.where(flat < 0, -k, k) + K).long()
    totH = totHh = 0.0
    for i in range(len(qd.mods)):
        s, n = qd.starts[i].item(), qd.sizes[i]
        cnt = torch.bincount(k[s:s + n], minlength=2 * K + 1).float()
        p = (cnt / cnt.sum()).cpu().numpy()
        nz = p[p > 0]
        totH += float(-(nz * np.log2(nz)).sum()) * n
        totHh += huff_avg(p) * n
    zf = float((k == K).sum()) / qd.n
    return totH / qd.n, totHh / qd.n, zf


# ------------------------------------------------------------------ model
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 4 * d, bias=False)
        self.fc2 = nn.Linear(4 * d, d, bias=False)
        self.h = h

    def forward(self, x, qd):
        B, T, C = x.shape
        z = self.ln1(x)
        qkv = F.linear(z, qd[self.qkv])
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.h, C // self.h).transpose(1, 2)
        k = k.view(B, T, self.h, C // self.h).transpose(1, 2)
        v = v.view(B, T, self.h, C // self.h).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).reshape(B, T, C)
        x = x + F.linear(a, qd[self.proj])
        h2 = self.ln2(x)
        return x + F.linear(F.gelu(F.linear(h2, qd[self.fc1])), qd[self.fc2])


class GPT(nn.Module):
    def __init__(self, vocab=256, d=160, L=4, h=4, ctx=128):
        super().__init__()
        self.te = nn.Embedding(vocab, d)
        self.pe = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(L)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.ctx = ctx

    def forward(self, x, qd):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.te(x) + self.pe(pos)[None]
        for b in self.blocks:
            h = b(h, qd)
        return F.linear(self.ln(h), qd[self.head])


def val_bpc(model, vs, qd):
    tot = 0.0
    for x, y in vs:
        logits = model(x, qd)
        tot += F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1)).item()
    return tot / len(vs) / math.log(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--d", type=int, default=160)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--qat", type=int, default=1)
    ap.add_argument("--dzlearn", type=int, default=0)
    ap.add_argument("--dzfix", type=float, default=0.0, help="freeze dz")
    ap.add_argument("--clipste", type=int, default=0)
    ap.add_argument("--anneal", type=float, default=0.0, help="λ 在总步数的前这个比例内线性升温")
    ap.add_argument("--qlr", type=float, default=1.0, help="量化器参数（step/ρ）学习率倍率")
    ap.add_argument("--cbr", type=float, default=0.0, help=">0 时启用码率控制：把 bpw 投影到目标值")
    ap.add_argument("--cbrlr", type=float, default=0.08, help="CBR 控制器增益")
    ap.add_argument("--cbrevery", type=int, default=20)
    ap.add_argument("--mode", default="wd", choices=["wd", "l1", "rate", "rate+dist"])
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--lamdist", type=float, default=0.0)
    ap.add_argument("--ratens", type=int, default=131072)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--save", type=int, default=1)
    args = ap.parse_args()

    raw = np.frombuffer(corpus(), dtype=np.uint8)
    split = int(len(raw) * 0.95)
    tr, va = raw[:split], raw[split:]
    vs = list(batches(va, args.ctx, 24, 24, seed=99))

    torch.manual_seed(args.seed)
    model = GPT(d=args.d, L=args.layers, ctx=args.ctx).to(DEV)
    qd = FlatQ([m for m in model.modules() if isinstance(m, nn.Linear)]).to(DEV)
    if args.dzfix > 0:
        with torch.no_grad():
            qd.rho.fill_(float(torch.log(torch.expm1(torch.tensor(args.dzfix - 1.0)))))
        qd.rho.requires_grad_(False)
        args.dzlearn = 1
    qp = [p for p in (qd.log_step, qd.rho) if p.requires_grad]
    if args.cbr > 0:
        qd.log_step.requires_grad_(False)   # 尺度交给控制器，不给梯度
        qp = [p for p in (qd.rho,) if p.requires_grad]
    opt = torch.optim.AdamW(
        [
            {"params": list(model.parameters()), "weight_decay": 0.01},
            {"params": qp, "weight_decay": 0.0, "lr": args.lr * args.qlr},
        ],
        lr=args.lr,
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr, args.lr * args.qlr], total_steps=args.steps, pct_start=0.1
    )

    fpw = {m: m.weight for m in qd.mods}
    t0 = time.time()
    for step, (x, y) in enumerate(batches(tr, args.ctx, args.bs, args.steps, seed=args.seed + 1)):
        if not args.qat:
            W = fpw
        elif args.dzlearn:
            W = quant_all(qd, True, bool(args.clipste))
        else:
            W = quant_all(qd, False, bool(args.clipste))
        logits = model(x, W)
        ce = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        loss = ce
        lam_t = args.lam
        if args.anneal > 0 and args.steps > 0:
            ramp = min(1.0, (step + 1) / max(1.0, args.anneal * args.steps))
            lam_t = args.lam * ramp
        if args.mode == "l1":
            loss = loss + args.lam * sum(m.weight.abs().sum() for m in qd.mods)
        if args.mode in ("rate", "rate+dist"):
            flat = catw(qd)
            step = qd.step()[qd.el]
            dzv = qd.dz()[qd.el] if args.dzlearn else torch.ones_like(step)
            nsel = min(args.ratens, flat.numel())
            if nsel < flat.numel():
                idx = torch.randint(0, flat.numel(), (nsel,), device=flat.device)
                fs, ss, ds = flat[idx], step[idx], dzv[idx]
            else:
                fs, ss, ds = flat, step, dzv
            loss = loss + lam_t * soft_rate(fs / ss, ds).mean()
            if args.mode == "rate+dist":
                q = dzq(fs, ss, ds, hard=False)
                loss = loss + args.lamdist * ((fs - q) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(qp, 1.0)
        opt.step()
        sched.step()
        if args.cbr > 0 and (step + 1) % args.cbrevery == 0:
            with torch.no_grad():
                b = measure_fast(qd, bool(args.dzlearn))
                # 码率对 step 单调递减：实测偏大就放大 step
                qd.gain.add_(args.cbrlr * (b - args.cbr) / max(args.cbr, 0.05))
                qd.gain.clamp_(-8.0, 8.0)

    model.eval()
    vb_fp = val_bpc(model, vs, fpw)
    W = quant_all(qd, True) if args.dzlearn else quant_all(qd, False)
    vb_q = val_bpc(model, vs, W)
    bits, hbits, zf = measure(qd, bool(args.dzlearn))

    if args.save:
        torch.save(
            {
                "model": {k: v.cpu() for k, v in model.state_dict().items()},
                "log_step": qd.log_step.detach().cpu(),
                "rho": qd.rho.detach().cpu(),
                "shapes": [tuple(s) for s in qd.shapes],
                "args": vars(args),
            },
            os.path.join(OUT, "models", f"{args.tag}_s{args.seed}.pt"),
        )

    row = (
        f"{args.tag},{args.mode},{args.dzlearn},{args.qat},{args.lam},{args.seed},"
        f"{vb_fp:.5f},{vb_q:.5f},{bits:.4f},{hbits:.4f},"
        f"{float(qd.dz().mean().detach()):.4f},{float(qd.step().mean().detach()):.5f},"
        f"{zf:.4f},{time.time()-t0:.1f}"
    )
    print("R," + row, flush=True)
    path = os.path.join(OUT, "runs.csv")
    if not os.path.exists(path):
        open(path, "w").write("tag,mode,dzlearn,qat,lam,seed,vb_fp,vb_q,bits,hbits,dz,step,zero,secs\n")
    open(path, "a").write(row + "\n")


if __name__ == "__main__":
    main()
