"""
analysis3.py — 修 Huffman + 两个新测量。

新测量：
  A  码流的自相关（lag 1..128）—— 有没有隐藏的周期性（存储布局/结构）
  B  按位置取模的熵 —— 有没有 stride 结构
  C  正确实现的 Huffman，给出真实的平均码长
"""

import heapq
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
                                off=int(d.get("offset")), size=int(d.get("size")))
fh = open(os.path.join(IR, "openvino_model.bin"), "rb")
mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)


def codes(nm):
    c = C[nm]
    r, cc = c["shape"]
    n = r * cc
    raw = np.frombuffer(mm, np.uint8, n // 2, c["off"])
    p = np.empty(n, np.uint8)
    p[0::2] = raw & 0x0F
    p[1::2] = raw >> 4
    return p.reshape(r, cc)


def ent(x):
    _, c = np.unique(x, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


# ---------------------------------------------------------------- Huffman
def huffman_lengths(cnt):
    h = []
    nid = 0
    for i in range(16):
        if cnt[i] > 0:
            h.append((cnt[i], nid, i))
            nid += 1
    heapq.heapify(h)
    parent = {}
    while len(h) > 1:
        a = heapq.heappop(h)
        b = heapq.heappop(h)
        new = nid
        nid += 1
        parent[a[1]] = new
        parent[b[1]] = new
        heapq.heappush(h, (a[0] + b[0], new, -1))
    leaves = [k for k in parent if k < 16] if False else None
    # 叶子 id 就是前 len(初始) 个 id
    depths = {}
    nleaf = sum(1 for i in range(16) if cnt[i] > 0)
    for i in range(16):
        if cnt[i] == 0:
            continue
        node = i
        d = 0
        while node in parent:
            node = parent[node]
            d += 1
        depths[i] = max(d, 1)
    return depths


def main():
    nm = "self.model.layers.18.mlp.up_proj.weight"
    W = codes(nm)
    print(f"{nm}  {W.shape}\n")

    # ---------------- C  正确 Huffman ----------------
    print("=" * 70)
    print("C  Canonical Huffman（修正版）")
    print("=" * 70)
    cnt = np.bincount(W.ravel(), minlength=16).astype(np.float64)
    p = cnt / cnt.sum()
    H = -float((p[p > 0] * np.log2(p[p > 0])).sum())
    d = huffman_lengths(cnt)
    avg = float(sum(p[s] * d[s] for s in d))
    kraft = sum(2.0 ** (-d[s]) for s in d)
    print(f"  H0           = {H:.4f}")
    print(f"  码长         = {dict(sorted(d.items()))}")
    print(f"  平均码长     = {avg:.4f}  （比熵多 {avg-H:+.4f}，比定长 4.0 省 {100*(1-avg/4):.1f}%）")
    print(f"  Kraft 和     = {kraft:.6f}  （必须 ≤ 1）")
    print(f"  最大码长     = {max(d.values())} → 单次 {1<<max(d.values())} 项 LUT 解码")

    # ---------------- A  自相关 ----------------
    print()
    print("=" * 70)
    print("A  码流自相关（lag 1..64，取 512 行的平均）")
    print("=" * 70)
    rows = W[rng_sel(W.shape[0], 512)].astype(np.float64)
    rows -= rows.mean(1, keepdims=True)
    rows /= rows.std(1, keepdims=True) + 1e-9
    ac = []
    for lag in range(1, 65):
        v = float((rows[:, :-lag] * rows[:, lag:]).mean())
        ac.append(v)
    ac = np.array(ac)
    print(f"  lag 1..8   : " + " ".join(f"{x:+.4f}" for x in ac[:8]))
    top = np.argsort(-np.abs(ac))[:5]
    print(f"  |ac| 最大的 5 个 lag: "
          + ", ".join(f"lag{int(t)+1}={ac[t]:+.4f}" for t in top))
    print(f"  |ac| 均值 = {np.abs(ac).mean():.5f}   理论噪声 ≈ {1/np.sqrt(W.shape[1]*512):.5f}")
    print(f"  → {'存在周期结构' if np.abs(ac).max() > 5/np.sqrt(W.shape[1]*512) else '无显著周期结构'}")

    # ---------------- B  位置取模熵 ----------------
    print()
    print("=" * 70)
    print("B  按位置取模的熵（检测 stride 结构）")
    print("=" * 70)
    for mod in (2, 4, 8, 16, 32, 64, 128):
        es = [ent(W[:, i::mod]) for i in range(mod)]
        print(f"  mod {mod:>3}: 熵 mean={np.mean(es):.4f} std={np.std(es):.4f} "
              f"max-min={np.max(es)-np.min(es):.4f}")


def rng_sel(n, k):
    return np.random.default_rng(0).choice(n, min(k, n), replace=False)


if __name__ == "__main__":
    main()
