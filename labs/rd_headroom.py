"""
rd_headroom.py — 把"还剩多少"算成一条精确的数字，而不是一句"还有 VQ"。

链条：
  1  全模型 252 个 i4 张量合并的**真实边际** p(code)
  2  从 p 反推 σ_code（模型无关：E[code²] = σ² + 1/12）
  3  当前设计（16 电平、步长=1、对称）的 (D, H) 精确值
  4  R(D) 界：0.5·log2(σ²/D)
  5  标量量化器族的可达 (D,H) 前沿 → 同样 D 下最优的标量设计能省多少
  6  格点 VQ（E8 / D4 / Leech）能再省多少
  7  每一步 → 精确 MiB

全部基于实测边际，不假设"标准高斯"以外的任何东西。
"""

import mmap
import os
import xml.etree.ElementTree as ET

import numpy as np

from paths import IR8B as IR
root = ET.parse(os.path.join(IR, "openvino_model.xml")).getroot()
C = {}
for layer in root.iter("layer"):
    if layer.get("type") != "Const":
        continue
    d = layer.find("data")
    if d is None:
        continue
    k = [x for x in (d.get("shape") or "").split(",") if x.strip()]
    C[layer.get("name")] = dict(shape=tuple(int(x) for x in k) if k else (1,),
                                et=d.get("element_type"),
                                off=int(d.get("offset", 0)), size=int(d.get("size")))
fh = open(os.path.join(IR, "openvino_model.bin"), "rb")
mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)

I4 = [nm for nm, c in sorted(C.items()) if c["et"] == "i4" and nm.endswith(".weight")]
N_I4 = sum(int(np.prod(C[nm]["shape"])) for nm in I4)

# ---------------------------------------------------------------- 1 实测边际
cnt = np.zeros(16, np.int64)
for nm in I4:
    c = C[nm]
    n = int(np.prod(c["shape"]))
    raw = np.frombuffer(mm, np.uint8, n // 2, c["off"])
    sym = np.empty(n, np.uint8)
    sym[0::2] = raw & 0x0F
    sym[1::2] = raw >> 4
    cnt += np.bincount(sym, minlength=16)
p = cnt / cnt.sum()
val = np.where(np.arange(16) < 8, np.arange(16), np.arange(16) - 16).astype(np.float64)
H0 = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
mean_code = float((p * val).sum())
E2 = float((p * val ** 2).sum())
var_code = E2 - mean_code ** 2
sigma_code = np.sqrt(max(var_code - 1.0 / 12, 1e-9))     # 去掉量化本身贡献的 1/12

print("=" * 78)
print("1  实测边际（252 个 i4 张量合并，%.3f B 参数）" % (N_I4 / 1e9))
print("=" * 78)
print("  code : " + " ".join(f"{v:>6.0f}" for v in val))
print("  P(%) : " + " ".join(f"{100*x:>6.2f}" for x in p))
print(f"\n  H0            = {H0:.4f} bit/权重")
print(f"  E[code]       = {mean_code:+.4f}    （理论 0；非零=量化器有直流偏置）")
print(f"  E[code²]      = {E2:.4f}")
print(f"  σ_code        = {sigma_code:.4f}    （去掉 1/12 量化方差后）")
print(f"  当前相对失真  = (1/12)/σ_code² = {1/(12*sigma_code**2):.6f}")

D_cur = 1.0 / (12.0 * sigma_code ** 2)          # D/σ²
R_D = 0.5 * np.log2(1.0 / D_cur)
print(f"  R(D) 界       = {R_D:.4f} bit/权重")
print(f"  距界          = {H0 - R_D:+.4f} bit   （空间填充损失上限 0.2546）")

# ---------------------------------------------------------------- 2 标量前沿
# 高斯源（归一化 σ=1）上扫 (步长 s, 电平数 L, 裁剪 c)，精确算 D 与 H
X = np.linspace(-8, 8, 800_001)
PDF = np.exp(-X * X / 2) / np.sqrt(2 * np.pi)
DX = X[1] - X[0]


def sq_curve(s, L):
    """步长 s，L 个电平，关于 0 对称（偶数 L 时偏移半个步长）。"""
    off = 0.0 if L % 2 else 0.5
    q = np.round(X / s - off) + off
    q = np.clip(q, -L // 2 + (0 if L % 2 else 0) - (0 if L % 2 else 1), L // 2 - (0 if L % 2 else 0))
    q = np.clip(q, -(L // 2), (L // 2))
    d = float((PDF * (X - q * s) ** 2).sum() * DX)
    codes, inv = np.unique(q, return_counts=True)
    pr = np.array([PDF[q == c].sum() * DX for c in codes])
    pr = pr / pr.sum()
    H = float(-(pr * np.log2(pr)).sum())
    return d, H, int(len(codes))


print()
print("=" * 78)
print("2  同样失真下，最优标量设计能省多少")
print("=" * 78)
best = None
for L in (12, 14, 16, 18, 20, 24, 32):
    for s in np.arange(0.35, 1.25, 0.01):
        d, H, k = sq_curve(s, L)
        if d <= D_cur:
            if best is None or H < best[1]:
                best = (d, H, L, s, k)
print(f"  当前设计      : D/σ²={D_cur:.5f}  H={H0:.4f}")
if best:
    d, H, L, s, k = best
    print(f"  最优标量      : D/σ²={d:.5f}  H={H:.4f}   电平={L} 步长={s:.3f}σ "
          f"（实际用到 {k} 个）")
    print(f"  → 同等失真下省 {H0 - H:+.4f} bit/权重")
else:
    print("  扫描没找到可行点")

# ---------------------------------------------------------------- 3 格点 VQ
print()
print("=" * 78)
print("3  换到矢量量化：格点的空间填充增益")
print("=" * 78)
G_CUBE = 1.0 / 12.0
LAT = [("Z^n（标量，现状）", 1.0 / 12.0),
       ("D4 棋盘格", 0.0766),
       ("E8", 0.0717),
       ("Λ16 (Barnes-Wall)", 0.0683),
       ("Λ24 (Leech)", 0.0658)]
for name, G in LAT:
    gain_db = 10 * np.log10(G_CUBE / G)
    print(f"  {name:<22} G={G:.5f}  颗粒增益 {gain_db:+.3f} dB  = {gain_db/6.0206:+.4f} bit")

# ---------------------------------------------------------------- 4 账本
print()
print("=" * 78)
print("4  账本（i4 部分 %.1f MiB 名义）" % (3312.0))
print("=" * 78)
MIB = 3312.0
rows = [("现状 4 bit 定长", 4.0),
        ("熵编码（已实现，QEC v1）", H0),
        ("  其中：最优标量 op-point", best[1] if best else H0),
        ("  其中：再叠 E8 格点", (best[1] if best else H0) - 0.108),
        ("  其中：再叠 Leech Λ24", (best[1] if best else H0) - 0.171),
        ("R(D) 界（不可达，需无穷维 VQ）", R_D)]
prev = None
for name, b in rows:
    mib = MIB * b / 4
    d = "" if prev is None else f"  ({100*(mib/prev-1):+.1f}%)"
    print(f"  {name:<34}{b:>7.4f} bit   {mib:>8.1f} MiB{d}")
    prev = mib
print()
print(f"  参照：u8 两块（embed + lm_head）1187.3 MiB，熵编码后 1069.9 MiB")
print(f"        当前全模型合计 3506.0 MiB（已 −22.1%）")
