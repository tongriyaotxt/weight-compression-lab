"""
analyze_entropy.py — 读 runs_entropy.csv，构造帕累托前沿，回答：
  在**相同任务质量**下，哪种正则能让模型更小？小多少？
"""

import csv
import sys
from collections import defaultdict

import numpy as np

from paths import HERE as _H
PATH = os.path.join(_H, "runs_entropy.csv")

rows = []
with open(PATH) as f:
    r = csv.reader(f)
    next(r)
    for a in r:
        if len(a) < 9:
            continue
        rows.append(dict(mode=a[0], lam=float(a[1]), seed=int(a[2]),
                         vbpc=float(a[3]), b16=float(a[4]), b8=float(a[5]),
                         b32=float(a[6]), zero=float(a[7]), secs=float(a[8])))

by = defaultdict(list)
for x in rows:
    key = (x["mode"], x["lam"])
    by[key].append(x)

print("=" * 86)
print("原始点（每格 3 或 2 个 seed 的均值±标准差）")
print("=" * 86)
print(f"  {'mode':<6}{'lam':>10}{'vbpc':>16}{'bits@16':>16}{'bits@8':>12}{'zero%':>8}")
print("  " + "-" * 82)
summary = []
for (mode, lam), xs in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1])):
    v = np.array([x["vbpc"] for x in xs])
    b = np.array([x["b16"] for x in xs])
    b8 = np.array([x["b8"] for x in xs])
    z = np.array([x["zero"] for x in xs])
    summary.append((mode, lam, v.mean(), v.std(), b.mean(), b.std(), b8.mean(), z.mean()))
    print(f"  {mode:<6}{lam:>10.4g}{v.mean():>10.4f}±{v.std():.3f}"
          f"{b.mean():>10.4f}±{b.std():.3f}{b8.mean():>12.4f}{100*z.mean():>8.2f}")

# ---------------- 每个 mode 的前沿 ----------------
front = defaultdict(list)
for s in summary:
    front[s[0]].append((s[2], s[4], s[1]))       # (vbpc, bits, lam)
for m in front:
    front[m].sort()


def interp_at(mode, q):
    pts = front[mode]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if q < min(xs) or q > max(xs):
        return None
    return float(np.interp(q, xs, ys))


base_v = np.mean([x["vbpc"] for x in by[("wd", 0.0)]])
base_b = np.mean([x["b16"] for x in by[("wd", 0.0)]])

print()
print("=" * 86)
print(f"同质量对比（基线 val = {base_v:.4f} bpc, bits@16 = {base_b:.4f}）")
print("=" * 86)
print(f"  {'mode':<8}{'bits@16':>10}{'省':>9}{'等效bpw':>10}{'需要lam':>12}")
print("  " + "-" * 50)
for m in ("l1", "rate", "hist"):
    b = interp_at(m, base_v)
    if b is None:
        print(f"  {m:<8}   该 mode 无法在基线质量处取值")
        continue
    lam = None
    for (mm, l), xs in by.items():
        if mm == m:
            vv = np.mean([x["vbpc"] for x in xs])
            if abs(vv - base_v) < 1e-9:
                lam = l
    print(f"  {m:<8}{b:>10.4f}{100*(1-b/base_b):>8.1f}%{b:>10.4f}  ")

print()
print("=" * 86)
print("质量预算 +0.10 bpc 时的体积（更现实的取舍）")
print("=" * 86)
q = base_v + 0.10
print(f"  {'mode':<8}{'bits@16':>10}{'体积比':>9}")
print("  " + "-" * 30)
for m in ("l1", "rate", "hist"):
    b = interp_at(m, q)
    if b is None:
        print(f"  {m:<8}   (超出该 mode 的扫描范围)")
        continue
    print(f"  {m:<8}{b:>10.4f}{b/base_b*100:>8.1f}%")

# ---------------- 体积换算 ----------------
n_params = 1.33e6
print()
print("=" * 86)
print("换算到这个模型（约 1.33M 参数，仅权重）")
print("=" * 86)
print(f"  基线 4-bit 定长 : {n_params*4/8/2**20:.2f} MiB")
print(f"  基线 + 熵编码   : {n_params*base_b/8/2**20:.2f} MiB")
for m in ("l1", "rate", "hist"):
    b = interp_at(m, base_v)
    if b:
        print(f"  {m} 同质量    : {n_params*b/8/2**20:.2f} MiB  ({100*(1-b/base_b):+.1f}%)")
