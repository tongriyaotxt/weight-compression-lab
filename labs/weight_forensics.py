"""
weight_forensics.py — 在真实的 INT4 Qwen3-8B 上做取证。

要回答的三件事：
  E0  4-bit 权重索引流的真实熵是多少？      → 决定"熵编码还能榨多少"
  E1  32x32 tile 的自相似性有多强？        → 决定"语法/程序化生成"成不成立
  E2  低秩 + tile 码本，谁更接近生成器？    → 决定"生成式权重"的最优点

全部基于 D:\\models\\npu-qwen3-8b 的 OpenVINO IR，直接读 .bin。
"""

import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np

from paths import IR8B as IR_DIR
XML = os.path.join(IR_DIR, "openvino_model.xml")
BIN = os.path.join(IR_DIR, "openvino_model.bin")

rng = np.random.default_rng(0)


# ---------------------------------------------------------------- IR 解析
def parse_consts(xml_path):
    root = ET.parse(xml_path).getroot()
    out = []
    for layer in root.iter("layer"):
        if layer.get("type") != "Const":
            continue
        d = layer.find("data")
        if d is None:
            continue
        parts = [x for x in (d.get("shape") or "").split(",") if x.strip() != ""]
        shape = tuple(int(x) for x in parts) if parts else (1,)
        out.append({
            "name": layer.get("name", ""),
            "shape": shape,
            "et": d.get("element_type"),
            "offset": int(d.get("offset")),
            "size": int(d.get("size")),
        })
    return out


def load_i4(mm, c):
    """OpenVINO i4：两个 4-bit 打包进一个字节，低 nibble 在前。"""
    rows, cols = c["shape"]
    nbytes = rows * cols // 2
    raw = np.frombuffer(mm, dtype=np.uint8, count=nbytes, offset=c["offset"])
    lo = raw & 0x0F
    hi = raw >> 4
    packed = np.empty(2 * nbytes, dtype=np.int8)
    packed[0::2] = lo
    packed[1::2] = hi
    return packed.reshape(rows, cols)


# ---------------------------------------------------------------- 熵
def entropy(counts):
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def h0(x):
    vals, cnt = np.unique(x, return_counts=True)
    return entropy(cnt.astype(np.float64)), dict(zip(vals.tolist(), cnt.tolist()))


def h1(x):
    """条件熵 H(x_t | x_{t-1})，按行展开。"""
    a = x[:, :-1].ravel().astype(np.int64)
    b = x[:, 1:].ravel().astype(np.int64)
    joint = np.zeros((16, 16))
    np.add.at(joint, (a, b), 1)
    total = joint.sum()
    px = joint.sum(1) / total
    out = 0.0
    for i in range(16):
        if joint[i].sum() > 0:
            out += px[i] * entropy(joint[i])
    return float(out)


# ---------------------------------------------------------------- tile
DIHEDRAL = [
    lambda t: t,
    lambda t: np.rot90(t, 1),
    lambda t: np.rot90(t, 2),
    lambda t: np.rot90(t, 3),
    lambda t: t.T,
    lambda t: np.rot90(t.T, 1),
    lambda t: np.rot90(t.T, 2),
    lambda t: np.rot90(t.T, 3),
]


def make_tiles(W, ts=32, limit=None):
    r, c = W.shape
    nr, nc = r // ts, c // ts
    T = W[:nr * ts, :nc * ts].reshape(nr, ts, nc, ts).transpose(0, 2, 1, 3)
    T = T.reshape(nr * nc, ts * ts).astype(np.float32)
    if limit and T.shape[0] > limit:
        idx = rng.choice(T.shape[0], limit, replace=False)
        T = T[idx]
    return T


def tile_codebook_gain(T, k=64, iters=8):
    """k-means 学一个 tile 码本，返回 (码本大小, 能量捕获率)。

    每个 tile 允许 8 种二面体变换，取最优匹配 —— 这就是"语法"假设的核心：
    如果权重里存在大量"同一模式的不同实例"，码本应该能用很小的 k 覆盖大部分能量。
    """
    N = T.shape[0]
    sub = T[rng.choice(N, min(4000, N), replace=False)]
    C = sub[rng.choice(sub.shape[0], k, replace=False)].copy()

    for _ in range(iters):
        # 最近邻（只用恒等变换，加速）
        d = (C ** 2).sum(1)[None, :] - 2 * T @ C.T + (T ** 2).sum(1)[:, None]
        lab = d.argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = T[m].mean(0)

    # 带 8 种变换的最优匹配
    best = np.full(N, np.inf)
    total_energy = (T ** 2).sum()
    for fn in DIHEDRAL:
        Cv = np.stack([fn(c.reshape(32, 32)).ravel() for c in C])
        d = (Cv ** 2).sum(1)[None, :] - 2 * T @ Cv.T + (T ** 2).sum(1)[:, None]
        best = np.minimum(best, d.min(1))
    resid = np.maximum(best, 0).sum()
    return k, 1.0 - resid / total_energy


def lowrank_gain(W, ranks=(4, 8, 16, 32, 64, 128, 256)):
    """对子矩阵做 SVD，返回每个 rank 的能量捕获率。"""
    A = W[:2048, :4096].astype(np.float32) if W.shape[0] >= 2048 else W.astype(np.float32)
    A = A - A.mean()
    # 用 Gram 矩阵避免大 SVD
    G = A @ A.T if A.shape[0] <= A.shape[1] else A.T @ A
    ev = np.linalg.eigvalsh(G.astype(np.float64))[::-1]
    ev = np.maximum(ev, 0)
    cum = np.cumsum(ev) / ev.sum()
    out = {}
    for r in ranks:
        out[r] = float(cum[min(r, len(cum)) - 1])
    return out


# ---------------------------------------------------------------- main
def main():
    consts = parse_consts(XML)
    by_name = {c["name"]: c for c in consts}

    # 选一个中层的 FFN 权重
    target = "self.model.layers.18.mlp.up_proj.weight"
    c = by_name[target]
    print(f"target: {c['name']}  shape={c['shape']}  et={c['et']}  size={c['size']/2**20:.1f} MiB")

    import mmap
    fh = open(BIN, "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)   # 只读，不占 4.7 GB 内存

    W = load_i4(mm, c)
    print(f"loaded: {W.shape}  dtype={W.dtype}  "
          f"min={W.min()} max={W.max()} mean={W.mean():+.3f} std={W.std():.3f}")

    hist = Counter(W.ravel().tolist())
    print("符号直方图（前 16）:", dict(sorted(hist.items())[:16]))
    print()

    # ---------------- E0: 熵 ----------------
    print("=" * 70)
    print("E0  4-bit 权重索引流的真实熵")
    print("=" * 70)
    e0, _ = h0(W)
    e1 = h1(W)
    print(f"  H0（全局）           : {e0:.4f} bits/权重")
    print(f"  H1（前一个符号条件） : {e1:.4f} bits/权重")
    print(f"  名义位宽             : 4.0000 bits/权重")
    print(f"  → 熵编码可省         : {100*(1-e0/4):.1f}% （H0）/ {100*(1-e1/4):.1f}% （H1）")
    print(f"  → 8B 模型 4.1 GB 权重 → {(1-e0/4)*4.1:.2f} GB")

    # 逐行熵（看行间差异）
    row_e = [h0(W[i])[0] for i in range(0, W.shape[0], W.shape[0] // 64)]
    print(f"  逐行 H0: mean={np.mean(row_e):.3f} std={np.std(row_e):.3f} "
          f"min={np.min(row_e):.3f} max={np.max(row_e):.3f}")
    print()

    # ---------------- E2: 低秩 ----------------
    print("=" * 70)
    print("E2  低秩结构")
    print("=" * 70)
    lr = lowrank_gain(W)
    for r, v in lr.items():
        print(f"  rank {r:>4}: 能量捕获 {100*v:6.2f}%   "
              f"（存储 {r*(W.shape[0]+W.shape[1])*2/8/2**20:.2f} MiB vs 原 {c['size']/2**20:.1f} MiB）")
    print()

    # ---------------- E1: tile 自相似 ----------------
    print("=" * 70)
    print("E1  32x32 tile 自相似性（语法假设）")
    print("=" * 70)
    T = make_tiles(W, ts=32, limit=4096)
    print(f"  tiles: {T.shape[0]} × 1024 权重")
    for k in (8, 32, 128):
        kk, gain = tile_codebook_gain(T, k=k, iters=6)
        print(f"  codebook k={kk:>4}: 能量捕获 {100*gain:6.2f}%  "
              f"（码本存储 {k*1024*1/2**10:.0f} KiB）")

    # 零假设：打乱位置，结构应当消失
    Ts = T.copy()
    for i in range(Ts.shape[0]):
        rng.shuffle(Ts[i])
    kk, gain_null = tile_codebook_gain(Ts, k=128, iters=6)
    print(f"  零假设（tile 内打乱）k=128: 能量捕获 {100*gain_null:6.2f}%   ← 这就是噪声地板")
    print()

    # ---------------- E3: scale 开销 ----------------
    print("=" * 70)
    print("E3  量化 scale 的额外开销")
    print("=" * 70)
    f16s = [x for x in consts if x["et"] == "f16"]
    print(f"  f16 张量数: {len(f16s)}")
    tot = sum(x["size"] for x in f16s)
    print(f"  f16 总字节: {tot/2**20:.1f} MiB")
    i4s = [x for x in consts if x["et"] == "i4"]
    i4t = sum(x["size"] for x in i4s)
    print(f"  i4  总字节: {i4t/2**20:.1f} MiB")
    print(f"  scale 占比: {100*tot/(tot+i4t):.2f}%  → 每权重额外 "
          f"{8*tot/(8.2e9):.3f} bit")
    for x in sorted(f16s, key=lambda z: -z["size"])[:5]:
        print(f"    {x['size']/2**20:8.3f} MiB  {x['shape']}  {x['name'][:60]}")


if __name__ == "__main__":
    main()
