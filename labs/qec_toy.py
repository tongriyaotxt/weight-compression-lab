"""
qec_toy.py — 把训练出来的模型真打成文件：code → rANS / Huffman → 实际字节。

前面所有 bits 都是"熵"，这篇文章的其余部分一直在说"理论 vs 文件还差多少"。
这个脚本把它闭合：对同一批码流，分别用
    · canonical Huffman（单表 / 逐张量表）
    · 静态 rANS（12-bit）
真编码一遍，报告实际字节数、每符号码长、以及和熵的差。
"""

import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dz_train import DEV, K, GPT, FlatQ, catw, huff_avg  # noqa: E402
from rans import build_table, encode, decode  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dz")


def codes_of(path):
    ck = torch.load(path, map_location=DEV, weights_only=False)
    model = GPT(d=160, L=4, ctx=128).to(DEV)
    model.load_state_dict(ck["model"])
    qd = FlatQ([m for m in model.modules() if isinstance(m, torch.nn.Linear)])
    qd.load_state_dict({"log_step": ck["log_step"], "rho": ck["rho"]}, strict=False)
    qd = qd.to(DEV)
    if not ck["args"].get("dzlearn", 0) and not ck["args"].get("dzfix", 0):
        qd.rho.data.fill_(-12.0)
    with torch.no_grad():
        flat = catw(qd).detach()
        step = qd.step()[qd.el]
        dzv = qd.dz()[qd.el]
        t = flat.abs() / step
        k = torch.clamp(torch.floor(t - dzv / 2.0) + 1.0, 0.0, float(K))
        k = (torch.where(flat < 0, -k, k) + K).long().cpu().numpy()
    slices = []
    for i in range(len(qd.mods)):
        s, n = qd.starts[i].item(), qd.sizes[i]
        slices.append(k[s:s + n])
    return slices


def entropy(syms, nsym=16):
    c = np.bincount(syms, minlength=nsym).astype(np.float64)
    p = c / c.sum()
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum()), p


def main():
    paths = sorted(glob.glob(os.path.join(OUT, "models", "*.pt")))
    sel = sys.argv[1:] or ["rate_dzL_0.01_s0", "rate_dz1_0.01_s0", "wdq_s0"]
    for tag in sel:
        hit = [p for p in paths if tag in os.path.basename(p)]
        if not hit:
            continue
        path = hit[0]
        slices = codes_of(path)
        n = sum(len(s) for s in slices)
        print("=" * 84)
        print(f"{os.path.basename(path)}   {n} 个符号，{len(slices)} 个张量")
        print("=" * 84)

        # 全局单表
        allsym = np.concatenate(slices)
        H, p = entropy(allsym)
        f, cum = build_table(p)
        blob, state = encode(allsym.tolist(), f, cum)
        back = decode(blob, state, f, cum, n)
        ok = np.array_equal(allsym, back)
        print(f"  单张全局表   H={H:5.3f}  rANS={len(blob)*8/n:5.3f} bit/符号"
              f"  Huffman={huff_avg(p):5.3f}  roundtrip={'OK' if ok else 'FAIL'}")

        # 逐张量表
        tot_bits = 0.0
        tot_bytes = 0
        tot_huf = 0.0
        allok = True
        for s in slices:
            Hs, ps = entropy(s)
            fs, cs = build_table(ps)
            b, st = encode(s.tolist(), fs, cs)
            bk = decode(b, st, fs, cs, len(s))
            allok &= bool(np.array_equal(s, bk))
            tot_bits += Hs * len(s)
            tot_bytes += len(b) + 4 + 15 * 2       # 表 + 状态
            tot_huf += huff_avg(ps) * len(s)
        print(f"  逐张量表     H={tot_bits/n:5.3f}  rANS={tot_bytes*8/n:5.3f} bit/符号"
              f"  Huffman={tot_huf/n:5.3f}  roundtrip={'OK' if allok else 'FAIL'}"
              f"  表开销={len(slices)*(15*2+4)} B")
        print(f"  → 4.5 MiB 级别对照：模型 {n/1e6:.2f}M 权重"
              f"  熵 {tot_bits/8/1024:.1f} KiB   rANS {tot_bytes/1024:.1f} KiB"
              f"   Huffman {tot_huf/8/1024:.1f} KiB")


if __name__ == "__main__":
    main()
