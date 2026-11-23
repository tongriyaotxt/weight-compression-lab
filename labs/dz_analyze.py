"""
dz_analyze.py — 汇总 labs/dz/runs.csv，并对保存的模型做"码流解剖"。

输出四块：
  1  按 tag 聚合（mean±std），按 bits 排序
  2  帕累托前沿（bits vs 部署后 val bpc）
  3  每个模型的逐张量步长分布（优化器自己做的 bit 分配，与"注水收益为零"对照）
  4  码字直方图（优化器实际发现的解长什么样：三元？稀疏？）
"""

import glob
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dz_train import DEV, K, GPT, FlatQ, catw  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dz")


def load_csv():
    rows = []
    with open(os.path.join(OUT, "runs.csv")) as f:
        hdr = f.readline().strip().split(",")
        for line in f:
            a = line.strip().split(",")
            if len(a) != len(hdr):
                continue
            d = dict(zip(hdr, a))
            for k in ("lam", "vb_fp", "vb_q", "bits", "hbits", "dz", "step", "zero", "secs"):
                d[k] = float(d[k])
            rows.append(d)
    return rows


def agg(rows):
    g = {}
    for r in rows:
        g.setdefault(r["tag"], []).append(r)
    out = []
    for tag, rs in g.items():
        m = lambda k: float(np.mean([x[k] for x in rs]))  # noqa: E731
        s = lambda k: float(np.std([x[k] for x in rs]))  # noqa: E731
        out.append({
            "tag": tag, "mode": rs[0]["mode"], "dzlearn": rs[0]["dzlearn"], "n": len(rs),
            "lam": rs[0]["lam"], "vb_q": m("vb_q"), "vb_q_sd": s("vb_q"),
            "bits": m("bits"), "bits_sd": s("bits"),
            "hbits": m("hbits"), "dz": m("dz"), "zero": m("zero"), "step": m("step"),
        })
    out.sort(key=lambda d: d["bits"])
    return out


def pareto(pts):
    """pts: (bits, bpc) -> 非支配集（bits 越小越好，bpc 越小越好）"""
    keep = []
    for p in sorted(pts, key=lambda z: z[0]):
        if not keep or p[1] < keep[-1][1] - 1e-9:
            keep.append(p)
    return keep


def rans_bits(p, table=4096):
    """12-bit 频率表下 rANS 的精确期望码长（每符号）。"""
    q = np.maximum(np.round(p * table).astype(np.int64), 1)
    q = q / q.sum()
    nz = p > 0
    return float(-(p[nz] * np.log2(q[nz])).sum())


def dump_model(path):
    ck = torch.load(path, map_location=DEV, weights_only=False)
    model = GPT(d=160, L=4, ctx=128).to(DEV)
    model.load_state_dict(ck["model"])
    model.eval()
    qd = FlatQ([m for m in model.modules() if isinstance(m, torch.nn.Linear)]).to(DEV)
    with torch.no_grad():
        qd.log_step.copy_(ck["log_step"].to(DEV))
        qd.rho.copy_(ck["rho"].to(DEV))
        flat = catw(qd).detach()
        step = qd.step()[qd.el]
        dzv = qd.dz()[qd.el]
        t = flat.abs() / step
        k = torch.clamp(torch.floor(t - dzv / 2.0) + 1.0, 0.0, float(K))
        k = (torch.where(flat < 0, -k, k) + K).long()
    dzz = float(qd.dz().mean().detach())
    print(f"\n--- {os.path.basename(path)}  dz={dzz:.3f}  step跨度={float(qd.step().max()/qd.step().min()):.2f}x ---")
    print("   #  shape              n      step     H     huff   rANS   0%   +1%   rail")
    tot = {"H": 0.0, "h": 0.0, "r": 0.0, "n": 0}
    for i in range(len(qd.mods)):
        s, n = qd.starts[i].item(), qd.sizes[i]
        cnt = torch.bincount(k[s:s + n], minlength=2 * K + 1).float()
        p = (cnt / cnt.sum()).cpu().numpy()
        nz = p[p > 0]
        H = float(-(nz * np.log2(nz)).sum())
        from dz_train import huff_avg
        tot["H"] += H * n
        tot["h"] += huff_avg(p) * n
        tot["r"] += rans_bits(p) * n
        tot["n"] += n
        rail = p[0] + p[-1]
        print(f"  {i:2d} {str(tuple(qd.shapes[i])):>14} {n:7d} {float(qd.step()[i].detach()):8.5f}"
              f" {H:5.3f} {huff_avg(p):6.3f} {rans_bits(p):6.3f} {p[K]*100:5.0f} {p[K+1]*100:5.0f} {rail*100:5.0f}")
    n = tot["n"]
    print(f"  合计: H={tot['H']/n:.3f}  huffman={tot['h']/n:.3f} (+{(tot['h']/tot['H']-1)*100:.0f}%)"
          f"  rANS={tot['r']/n:.3f} (+{(tot['r']/tot['H']-1)*100:.1f}%)")


def main():
    rows = load_csv()
    A = agg(rows)
    print("=" * 96)
    print(f"{'tag':<18}{'mode':<10}{'dzL':>4}{'n':>3}{'lam':>9}{'bits':>16}{'vb_q':>16}{'dz':>7}{'zero':>7}")
    print("=" * 96)
    for d in A:
        print(f"{d['tag']:<18}{d['mode']:<10}{d['dzlearn']:>4}{d['n']:>3}{d['lam']:>9.5g}"
              f"{d['bits']:>9.3f}±{d['bits_sd']:<6.3f}{d['vb_q']:>9.3f}±{d['vb_q_sd']:<6.3f}"
              f"{d['dz']:>7.2f}{d['zero']*100:>6.1f}%")

    print("\n" + "=" * 60)
    print("帕累托前沿（bits vs 部署后 val bpc，越小越好）")
    print("=" * 60)
    pts = [(d["bits"], d["vb_q"], d["tag"]) for d in A]
    for b, q, t in pareto(pts):
        print(f"  {b:6.3f} bits   {q:6.3f} bpc   {t}")

    # 等质量比较：基线 wdq 的质量下，谁能用更少 bits
    base = [d for d in A if d["tag"].startswith("wdq")]
    if base:
        q0 = base[0]["vb_q"]
        print(f"\n等质量比较（vb_q <= {q0:.3f}，即 QAT-INT4 基线质量）")
        for d in A:
            if d["vb_q"] <= q0 + 0.005:
                print(f"  {d['tag']:<20} {d['bits']:5.3f} bits   {d['vb_q']:5.3f} bpc   "
                      f"相对 4-bit 省 {(1 - d['bits']/3.907)*100:5.1f}%")

    print("\n" + "=" * 60)
    print("模型解剖")
    print("=" * 60)
    for p in sorted(glob.glob(os.path.join(OUT, "models", "*.pt"))):
        if any(k in os.path.basename(p) for k in ("wdq_s0", "l1_dzL_3e-06_s0", "rate_dz1_0.01_s0", "rate_dzL_0.003_s0")):
            dump_model(p)


if __name__ == "__main__":
    main()
