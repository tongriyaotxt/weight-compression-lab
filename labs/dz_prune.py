"""
dz_prune.py — 后验降档：不改 step、不改 dz，只把码字表**嵌套地收窄**。

为什么不能用"改 step"来降档：
  量化器的重建值是 c·step（或死区变体）。改 step 就等于整体缩放权重；
  更致命的是把 step 减小会让原本落在死区里的 78% 权重突然变成 ±1 ——
  零类会被翻掉。所以 step 是"尺度"，不是"档位旋钮"。

正确的嵌套旋钮是**字母表**：
  codes ±1..±7 保留，但把 |c| > r 的码合并到 ±r。
  重建值一个都不变，只是可用符号数从 15 降到 2r+1。
  于是同一个模型可以出 r=7,5,3,1 等多个体积档，且码流熵单调下降。

这一份测的正是"后验能做到多少"，用来和"在该码率下重新训练"（dz_train）对照：
  两者之差 = 训练时把码率放进目标函数的真实价值。
"""

import glob
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dz_train import DEV, K, GPT, FlatQ, batches, catw, corpus, huff_avg  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dz")


def huff(p):
    return huff_avg(p)


@torch.no_grad()
def run(path, vs):
    ck = torch.load(path, map_location=DEV, weights_only=False)
    model = GPT(d=160, L=4, ctx=128).to(DEV)
    model.load_state_dict(ck["model"])
    model.eval()
    qd = FlatQ([m for m in model.modules() if isinstance(m, torch.nn.Linear)]).to(DEV)
    qd.log_step.copy_(ck["log_step"].to(DEV))
    qd.rho.copy_(ck["rho"].to(DEV))
    if not ck["args"].get("dzlearn", 0) and not ck["args"].get("dzfix", 0):
        qd.rho.fill_(-12.0)  # 训练时 dz 恒为 1

    flat = catw(qd)
    step = qd.step()[qd.el]
    dzv = qd.dz()[qd.el]
    t = flat.abs() / step
    c = torch.clamp(torch.floor(t - dzv / 2.0) + 1.0, 0.0, float(K))
    c = torch.where(flat < 0, -c, c)

    print(f"\n=== {os.path.basename(path)}  dz={float(qd.dz().mean()):.3f} ===")
    print(f"  {'r':>2}{'符号数':>6}{'bits':>9}{'huff':>8}{'bpc':>9}   说明")
    for r in range(K, 0, -1):
        cr = torch.clamp(c, -float(r), float(r))
        idx = (cr + K).long()
        tot = totH = 0.0
        for i in range(len(qd.mods)):
            s, n = qd.starts[i].item(), qd.sizes[i]
            cnt = torch.bincount(idx[s:s + n], minlength=2 * K + 1).float()
            p = (cnt / cnt.sum()).cpu().numpy()
            nz = p[p > 0]
            tot += float(-(nz * np.log2(nz)).sum()) * n
            totH += huff(p) * n
        bits, hbits = tot / qd.n, totH / qd.n
        qflat = torch.sign(cr) * torch.where(
            cr == 0, torch.zeros_like(cr), torch.clamp(cr.abs() - 0.5 + dzv / 2.0, min=0.0) * step)
        W = {}
        for i, m in enumerate(qd.mods):
            s, n = qd.starts[i].item(), qd.sizes[i]
            W[m] = qflat[s:s + n].view(qd.shapes[i])
        v = 0.0
        for x, y in vs:
            v += F.cross_entropy(model(x, W).reshape(-1, 256), y.reshape(-1)).item()
        print(f"  {r:>2}{2*r+1:>6}{bits:>9.3f}{hbits:>8.3f}{v/len(vs)/math.log(2):>9.3f}"
              f"   {'等价 3-bit' if r==3 else '等价 2-bit' if r==1 else '等价 %d-bit' % math.ceil(math.log2(2*r+1))}")


def main():
    raw = np.frombuffer(corpus(), dtype=np.uint8)
    va = raw[int(len(raw) * 0.95):]
    vs = list(batches(va, 128, 24, 24, seed=99))
    sel = [a for a in sys.argv[1:]]
    paths = sorted(glob.glob(os.path.join(OUT, "models", "*.pt")))
    if sel:
        paths = [p for p in paths if any(a in os.path.basename(p) for a in sel)]
    for p in paths:
        run(p, vs)


if __name__ == "__main__":
    main()
