"""
alloc.py — 基于"权重是 i.i.d."这个实测事实，做一次真正的速率分配。

推导（这是本轮的核心）：
  线性层 y = Wx，量化误差 ΔW 的元素 i.i.d.，方差 D。
  E‖Δy‖² = tr(E[ΔWᵀΔW] · E[xxᵀ]) = D · E‖x‖²        （因为 E[ΔWᵀΔW] ≈ D·I）

  所以逐层灵敏度 = D_t × a_t，其中 a_t = E‖x_t‖² 是**一个标量**。
  Hessian 那一整套机器（HAWQ/GPTQ 的 XXᵀ 估计）在这里退化成一行。

  分配问题：  min Σ D_t·a_t   s.t.  Σ n_t·R_t = B
  拉格朗日 →  R_t = 0.5·log2(σ_t²·a_t/θ)，  D_t = θ/(σ... ) 即反向注水。
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
                                off=int(d.get("offset")), size=int(d.get("size")))
fh = open(os.path.join(IR, "openvino_model.bin"), "rb")
mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)

CLIP_SIGMA = 4.08   # max 启发式裁剪点


def table():
    """返回 L -> (D/σ², H) 的对照表（高斯 + max 裁剪）。"""
    x = np.linspace(-6, 6, 600001)
    pdf = np.exp(-x * x / 2)
    pdf /= pdf.sum()
    out = {}
    for L in (2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256):
        st = 2 * CLIP_SIGMA / L
        q = np.clip(np.round(x / st), -L // 2, L // 2 - 1).astype(np.int64)
        p = np.bincount(q - q.min(), weights=pdf)
        p = p / p.sum()
        H = -float((p[p > 0] * np.log2(p[p > 0])).sum())
        D = float((pdf * (x - q * st) ** 2).sum())
        out[L] = (D, H, st)
    return out


def main():
    T = table()
    print("=" * 74)
    print("0  电平数 → (失真, 熵) 表（高斯，max 裁剪 ±4.08σ）")
    print("=" * 74)
    print(f"  {'L':>5}{'D/σ²':>12}{'H(bits)':>10}{'step/σ':>10}")
    for L, (D, H, st) in T.items():
        print(f"  {L:>5}{D:>12.6f}{H:>10.3f}{st:>10.4f}")

    # ---------------- 收集每个张量的 (n, σ) ----------------
    items = []
    for nm, c in sorted(C.items()):
        if c["et"] != "i4":
            continue
        if not nm.endswith(".weight"):
            continue
        sc = C.get(nm + "/scale")
        if sc is None:
            continue
        s = np.frombuffer(mm, np.float16, int(np.prod(sc["shape"])), sc["off"])
        s = np.abs(s.astype(np.float32))
        sigma = float(np.median(s)) / (CLIP_SIGMA / 8)      # step = max/8 = 4.08σ/8
        items.append((nm, int(np.prod(c["shape"])), sigma, float(np.median(s))))

    n_tot = sum(i[1] for i in items)
    print()
    print("=" * 74)
    print(f"1  张量统计：{len(items)} 个 i4 权重，{n_tot/1e9:.3f} B 参数")
    print("=" * 74)
    sig = np.array([i[2] for i in items])
    print(f"  σ 范围: {sig.min():.5f} .. {sig.max():.5f}   （比值 {sig.max()/sig.min():.2f}×）")
    print(f"  D ∝ σ²，所以失真敏感度跨 {sig.max()**2/sig.min()**2:.1f}×")

    # ---------------- 基线：全部 L=16 ----------------
    Ls = [4, 5, 6, 8, 12, 16, 24, 32]
    base_bits = n_tot * T[16][1]
    base_D = sum(n * s * s * T[16][0] for _, n, s, _ in items)
    print()
    print(f"  基线（全部 L=16）: {base_bits/8/2**20:.1f} MiB, "
          f"总失真 {base_D:.6e}")

    # ---------------- 最优分配 @ 同样的比特预算 ----------------
    print()
    print("=" * 74)
    print("2  同样比特预算下的最优分配（反向注水，a_t=1）")
    print("=" * 74)
    lo, hi = 1e-12, 1e6
    for _ in range(200):
        theta = (lo * hi) ** 0.5
        bits = 0.0
        D = 0.0
        for _, n, s, _ in items:
            # 每个张量选使 "D·σ² + θ·H" 最小的 L
            best = None
            for L in Ls:
                Dl, Hl, _ = T[L]
                cost = s * s * Dl + theta * Hl
                if best is None or cost < best[0]:
                    best = (cost, L)
            L = best[1]
            bits += n * T[L][1]
            D += n * s * s * T[L][0]
        if bits > base_bits:
            lo = theta
        else:
            hi = theta
    print(f"  最优总失真 {D:.6e}   vs 基线 {base_D:.6e}")
    print(f"  → 同体积下质量提升 {10*np.log10(base_D/D):+.2f} dB")
    print(f"  → 等效多 {0.5*np.log2(base_D/D):+.3f} bit")
    from collections import Counter
    cntL = Counter()
    for _, n, s, _ in items:
        best = None
        for L in Ls:
            Dl, Hl, _ = T[L]
            cost = s * s * Dl + theta * Hl
            if best is None or cost < best[0]:
                best = (cost, L)
        cntL[best[1]] += n
    print(f"  分配结果: " + ", ".join(f"L={L}: {v/1e9:.2f}B" for L, v in sorted(cntL.items())))

    # ---------------- 固定 3.0 bpw 时的最优分配 ----------------
    print()
    print("=" * 74)
    print("3  固定 3.0 bpw（30B 落到 11.3 GB）时的分配")
    print("=" * 74)
    target = n_tot * 3.0
    lo, hi = 1e-12, 1e6
    for _ in range(200):
        theta = (lo * hi) ** 0.5
        bits = 0.0
        D = 0.0
        for _, n, s, _ in items:
            best = None
            for L in Ls:
                Dl, Hl, _ = T[L]
                cost = s * s * Dl + theta * Hl
                if best is None or cost < best[0]:
                    best = (cost, L)
            L = best[1]
            bits += n * T[L][1]
            D += n * s * s * T[L][0]
        if bits > target:
            lo = theta
        else:
            hi = theta
    uniform3 = sum(n * s * s * T[8][0] for _, n, s, _ in items)
    print(f"  最优：失真 {D:.6e}")
    print(f"  朴素（全 L=8）：失真 {uniform3:.6e}")
    print(f"  → 反向注水比【一刀切 3 bit】好 {10*np.log10(uniform3/D):+.2f} dB")


if __name__ == "__main__":
    main()
