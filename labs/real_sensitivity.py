"""
real_sensitivity.py — Qwen3-1.7B 的逐层量化敏感性 + 混合精度分配。

背景：这个 IR 只有 INT4 结果，没有 FP16 原件，所以"再压一点"的 PTQ 是无效实验
（第二次量化叠在已经贴边的网格上，ppl 直接炸到 1e5）。但有一件事可以测：
**哪些层碰不得**。

做法（一次载入，逐层量化再还原）：
  1  对每层单独用 s 倍步长量化，测 ppl 变化 → 敏感性地图
  2  只把"不敏感"的层压到低码率，敏感层保持原网格 → 总 bpw / ppl

这直接检验上一轮的负结果："逐层分配收益为零，因为权重是 i.i.d."。
如果敏感性地图是尖的，那个结论就要改。
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_qwen import DEV, Tok, load, ppl  # noqa: E402

K = 7
NAMES = ("q", "k", "v", "o", "gate", "up", "down")


@torch.no_grad()
def q_row(w, s, dz=1.0):
    out, _ = w.shape
    r = w.reshape(out, -1).float()
    sig = r.std(dim=1, keepdim=True).clamp(min=1e-8)
    step = s * sig
    t = r.abs() / step
    k = torch.clamp(torch.floor(t - dz / 2.0) + 1.0, 0.0, float(K))
    mag = torch.where(k > 0, (dz / 2.0 + k - 0.5) * step, torch.zeros_like(t))
    return (torch.sign(r) * mag).reshape(w.shape)


@torch.no_grad()
def bits_of(w, s):
    rec = q_row(w, s)
    step = s * w.reshape(w.shape[0], -1).float().std(dim=1, keepdim=True).clamp(min=1e-8)
    t = w.reshape(w.shape[0], -1).float().abs() / step
    k = torch.clamp(torch.floor(t - 0.5) + 1.0, 0.0, float(K))
    k = (torch.where(w.reshape(w.shape[0], -1) < 0, -k, k) + K).long().reshape(-1)
    cnt = torch.bincount(k, minlength=2 * K + 1).float()
    p = (cnt / cnt.sum())[cnt > 0]
    return float(-(p * torch.log2(p)).sum())


def main():
    tok = Tok()
    m, _, _ = load()
    m = m.to(DEV).to(torch.bfloat16)
    p0, _ = ppl(m, tok)
    print(f"基线 ppl = {p0:.3f}\n")

    S = 1.2          # 激进档：约 1.8 bpw
    rows = []
    for i, l in enumerate(m.layers):
        backup = {nm: getattr(l, nm).weight.data.clone() for nm in NAMES}
        for nm in NAMES:
            mod = getattr(l, nm)
            mod.weight.data.copy_(q_row(mod.weight.data, S).to(torch.bfloat16))
        p, _ = ppl(m, tok)
        for nm in NAMES:
            getattr(l, nm).weight.data.copy_(backup[nm])
        rows.append((i, p))
        print(f"  layer {i:2d} 单独量化@1.8bpw  ppl = {p:12.2f}   倍率 {p/p0:10.1f}x", flush=True)

    rows.sort(key=lambda z: z[1])
    safe = [i for i, p in rows if p < p0 * 1.2]
    print(f"\n安全层（ppl 涨幅 <20%）：{safe}")

    # 混合精度：安全层压到 S，其余保持原网格
    m2, _, _ = load()
    m2 = m2.to(DEV).to(torch.bfloat16)
    tb = tn = 0
    for i, l in enumerate(m2.layers):
        for nm in NAMES:
            w = getattr(l, nm).weight.data
            s = S if i in safe else 0.4934
            tb += bits_of(w, s) * w.numel()
            tn += w.numel()
            if i in safe:
                getattr(l, nm).weight.data.copy_(q_row(w, S).to(torch.bfloat16))
    p, _ = ppl(m2, tok)
    print(f"\n混合精度（{len(safe)}/{len(m2.layers)} 层压到 {S}）")
    print(f"  平均 bpw = {tb/tn:.3f}   （原网格 ≈2.96）")
    print(f"  ppl = {p:.2f}  （基线 {p0:.2f}，{p/p0-1:+.1%}）")

    json.dump(dict(baseline=p0, s=S, per_layer=rows, safe=safe,
                   mixed_bpw=tb / tn, mixed_ppl=p),
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dz",
                                "real_sensitivity.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
