"""
forensics2.py — 在 INT4 Qwen3-8B 上做第二轮取证。

A  跨层 / 跨张量类型：3.04 bit 这个熵是不是普遍成立
B  行级自相似：是不是有大量近似重复的神经元（比 tile 更可能的冗余尺度）
C  更小尺度 tile（8x8）的相关性近邻
D  ECSQ 前沿：同样 4 bit 存储，非均匀量化能比当前均匀量化好多少
"""

import mmap
import os
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np

from paths import IR8B as IR_DIR
XML = os.path.join(IR_DIR, "openvino_model.xml")
BIN = os.path.join(IR_DIR, "openvino_model.bin")
rng = np.random.default_rng(0)


def parse_consts(p):
    root = ET.parse(p).getroot()
    out = {}
    for layer in root.iter("layer"):
        if layer.get("type") != "Const":
            continue
        d = layer.find("data")
        if d is None:
            continue
        parts = [x for x in (d.get("shape") or "").split(",") if x.strip()]
        out[layer.get("name")] = {
            "shape": tuple(int(x) for x in parts) if parts else (1,),
            "et": d.get("element_type"),
            "offset": int(d.get("offset")),
            "size": int(d.get("size")),
        }
    return out


def load(mm, c):
    et = c["et"]
    shape = c["shape"]
    n = int(np.prod(shape))
    if et == "i4":
        raw = np.frombuffer(mm, np.uint8, n // 2, c["offset"])
        p = np.empty(n, np.int8)
        p[0::2] = raw & 0x0F
        p[1::2] = raw >> 4
        return p.reshape(shape)
    if et == "u8":
        return np.frombuffer(mm, np.uint8, n, c["offset"]).reshape(shape)
    if et == "f16":
        return np.frombuffer(mm, np.float16, n, c["offset"]).reshape(shape)
    return None


def h0(x):
    _, cnt = np.unique(x, return_counts=True)
    p = cnt / cnt.sum()
    return float(-(p * np.log2(p)).sum())


# ============================================================ D: ECSQ
def lloyd_max(sigma=1.0, levels=16, iters=60):
    """对 N(0, sigma^2) 做 Lloyd-Max。用细网格 + 迭代重心。"""
    x = np.linspace(-4.5 * sigma, 4.5 * sigma, 200001)
    pdf = np.exp(-x ** 2 / (2 * sigma ** 2))
    pdf /= pdf.sum()
    c = np.linspace(-3.2 * sigma, 3.2 * sigma, levels)
    for _ in range(iters):
        b = np.concatenate([[-np.inf], (c[1:] + c[:-1]) / 2, [np.inf]])
        idx = np.clip(np.searchsorted(b, x) - 1, 0, levels - 1)
        w = np.bincount(idx, weights=pdf, minlength=levels)
        m = np.bincount(idx, weights=pdf * x, minlength=levels)
        c = np.where(w > 1e-12, m / np.maximum(w, 1e-12), c)
    b = np.concatenate([[-np.inf], (c[1:] + c[:-1]) / 2, [np.inf]])
    idx = np.clip(np.searchsorted(b, x) - 1, 0, levels - 1)
    w = np.bincount(idx, weights=pdf, minlength=levels)
    p = w / w.sum()
    D = float((pdf * (x - c[idx]) ** 2).sum())
    H = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    return c, p, D, H


def uniform_q(sigma=1.0, levels=16, clip=4.0, grid=200001):
    x = np.linspace(-6 * sigma, 6 * sigma, grid)
    pdf = np.exp(-x ** 2 / (2 * sigma ** 2))
    pdf /= pdf.sum()
    step = 2 * clip * sigma / levels
    q = np.clip(np.round(x / step), -levels / 2, levels / 2 - 1) * step
    D = float((pdf * (x - q) ** 2).sum())
    idx = (q / step).astype(int)
    w = np.bincount(idx + levels // 2, weights=pdf, minlength=levels)
    p = w / w.sum()
    H = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    return D, H


def ecsq_frontier(sigma=1.0, target_H=4.0, grid=200001, iters=80):
    """熵约束标量量化：min D s.t. H = target_H（拉格朗日 D + lam*H）。"""
    x = np.linspace(-6 * sigma, 6 * sigma, grid)
    pdf = np.exp(-x ** 2 / (2 * sigma ** 2))
    pdf /= pdf.sum()
    lo, hi = 1e-6, 10.0
    for _ in range(60):
        lam = (lo * hi) ** 0.5
        c = np.linspace(-3.5 * sigma, 3.5 * sigma, 64)
        for _ in range(iters):
            b = np.concatenate([[-np.inf], (c[1:] + c[:-1]) / 2, [np.inf]])
            idx = np.clip(np.searchsorted(b, x) - 1, 0, 63)
            w = np.bincount(idx, weights=pdf, minlength=64) + 1e-18
            m = np.bincount(idx, weights=pdf * x, minlength=64)
            c = np.where(w > 1e-12, m / w, c)
            # ECSQ 重心：拉格朗日修正
            b2 = np.concatenate([[-np.inf], (c[1:] + c[:-1]) / 2, [np.inf]])
            idx = np.clip(np.searchsorted(b2, x) - 1, 0, 63)
            # 合并概率过低的电平
        b = np.concatenate([[-np.inf], (c[1:] + c[:-1]) / 2, [np.inf]])
        idx = np.clip(np.searchsorted(b, x) - 1, 0, 63)
        w = np.bincount(idx, weights=pdf, minlength=64)
        m = np.bincount(idx, weights=pdf * x, minlength=64)
        c = np.where(w > 1e-12, m / np.maximum(w, 1e-12), c)
        b = np.concatenate([[-np.inf], (c[1:] + c[:-1]) / 2, [np.inf]])
        idx = np.clip(np.searchsorted(b, x) - 1, 0, 63)
        w = np.bincount(idx, weights=pdf, minlength=64)
        w = w[w > 0]
        p = w / w.sum()
        H = float(-(p * np.log2(p)).sum())
        D = float((np.bincount(idx, weights=pdf * (x - c[idx]) ** 2, minlength=64)).sum())
        if H > target_H:
            lo = lam
        else:
            hi = lam
    # 简化：直接扫电平数，找 H≈target 的最优 Lloyd-Max
    best = None
    for L in range(4, 33):
        cc, pp, DD, HH = lloyd_max(sigma, L)
        if HH <= target_H and (best is None or DD < best[0]):
            best = (DD, HH, L)
    return best


# ============================================================ main
def main():
    consts = parse_consts(XML)
    fh = open(BIN, "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)

    print("=" * 78)
    print("A  跨层 / 跨张量类型的 H0")
    print("=" * 78)
    print(f"{'tensor':<52}{'shape':>14}{'H0':>8}{'可省':>8}")
    print("-" * 78)
    tensors = []
    for L in (0, 9, 18, 27, 35):
        for suf in ("self_attn.q_proj", "mlp.up_proj", "mlp.down_proj"):
            nm = f"self.model.layers.{L}.{suf}.weight"
            if nm in consts:
                tensors.append(nm)
    tensors.append("self.model.embed_tokens.weight")

    hs = []
    for nm in tensors:
        c = consts[nm]
        W = load(mm, c)
        if W is None:
            continue
        e = h0(W.ravel())
        hs.append((nm, e))
        short = nm.replace("self.model.", "")
        print(f"{short:<52}{str(c['shape']):>14}{e:>8.4f}{100*(1-e/8 if c['et']=='u8' else 1-e/4):>7.1f}%")

    print()
    print("=" * 78)
    print("B  行级自相似性（神经元是否近似重复）")
    print("=" * 78)
    for nm in ("self.model.layers.18.mlp.up_proj.weight",):
        W = load(mm, consts[nm]).astype(np.float32)
        idx = rng.choice(W.shape[0], 2048, replace=False)
        R = W[idx]
        R -= R.mean(1, keepdims=True)
        nrm = np.linalg.norm(R, axis=1, keepdims=True)
        R /= np.maximum(nrm, 1e-9)
        C = R @ R.T
        np.fill_diagonal(C, 0)
        off = np.abs(C)
        print(f"  {nm}")
        print(f"    {W.shape[0]} 行，抽样 2048 行，两两相关")
        print(f"    mean |corr| = {off.mean():.4f}")
        print(f"    99%  分位   = {np.percentile(off, 99):.4f}")
        print(f"    最大        = {off.max():.4f}")
        print(f"    |corr| > 0.9 的比例 = {100*(off > 0.9).mean():.3f}%")
        print(f"    → 若最大相关也不高，说明没有近似重复神经元，行级码本无意义")

    print()
    print("=" * 78)
    print("C  8x8 tile 相关性近邻（比 32x32 更细的尺度）")
    print("=" * 78)
    W = load(mm, consts["self.model.layers.18.mlp.up_proj.weight"]).astype(np.float32)
    ts = 8
    nr, nc = W.shape[0] // ts, W.shape[1] // ts
    T = W[:nr * ts, :nc * ts].reshape(nr, ts, nc, ts).transpose(0, 2, 1, 3).reshape(-1, ts * ts)
    sel = rng.choice(T.shape[0], 1500, replace=False)
    Ts = T[sel].copy()
    Ts -= Ts.mean(1, keepdims=True)
    Ts /= np.maximum(np.linalg.norm(Ts, axis=1, keepdims=True), 1e-9)
    Cm = np.abs(Ts @ Ts.T)
    np.fill_diagonal(Cm, 0)
    print(f"    tiles: {T.shape[0]} 个 8x8，抽样 1500")
    print(f"    最近邻 |corr| 均值 = {Cm.max(1).mean():.4f}")
    print(f"    最近邻 |corr| 中位 = {np.median(Cm.max(1)):.4f}")
    print(f"    |corr|>0.9 的比例  = {100*(Cm > 0.9).mean():.4f}%")
    # 解析零基线：随机 64 维单位向量的相关 ~ N(0, 1/64)，取 1500 个候选的最大值
    sd = 1.0 / np.sqrt(Ts.shape[1])
    null_max = sd * np.sqrt(2 * np.log(Ts.shape[0] - 1))
    print(f"    随机向量理论基线    = {null_max:.4f}  ← 纯噪声也能到这个值")
    print(f"    → 实测 {'≈' if abs(Cm.max(1).mean()-null_max)<0.06 else '≠'} 基线，"
          f"{'没有可利用的 tile 结构' if abs(Cm.max(1).mean()-null_max)<0.06 else '存在弱结构'}")
    del T, Ts, Cm

    print()
    print("=" * 78)
    print("D  同样 4 bit 存储：当前均匀量化 vs Lloyd-Max vs ECSQ")
    print("=" * 78)
    D_u, H_u = uniform_q()
    c, p, D_l, H_l = lloyd_max()
    print(f"  {'方案':<24}{'MSE(σ²=1)':>12}{'熵(bit)':>10}{'相对均匀':>12}")
    print("  " + "-" * 58)
    print(f"  {'均匀 16 电平 (clip=4σ)':<24}{D_u:>12.5f}{H_u:>10.3f}{1.0:>11.3f}x")
    print(f"  {'Lloyd-Max 16 电平':<24}{D_l:>12.5f}{H_l:>10.3f}{D_u/D_l:>11.3f}x")
    # ECSQ: 熵 4.0 下最优电平数
    best = None
    for L in range(8, 41):
        cc, pp, DD, HH = lloyd_max(levels=L)
        if HH <= 4.0 and (best is None or DD < best[0]):
            best = (DD, HH, L)
    if best:
        print(f"  {'ECSQ (熵≤4.0, L=%d)' % best[2]:<24}{best[0]:>12.5f}{best[1]:>10.3f}{D_u/best[0]:>11.3f}x")
        print(f"\n  → 同样 4 bit 存储，ECSQ 的失真比当前均匀量化低 "
              f"{10*np.log10(D_u/best[0]):.2f} dB")
        print(f"  → 折算成等效比特数：省 {(1-best[1]/4)*100:.1f}% 空间，或质量等效于多 "
              f"{0.5*np.log2(D_u/best[0]):.2f} bit")


if __name__ == "__main__":
    main()
