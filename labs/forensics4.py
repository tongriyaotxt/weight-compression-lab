"""
forensics4.py — 修正版：反量化真实权重，量出"max 裁剪"到底浪费多少。

上一版 bug：W 和 scale 抽了不同的随机行。这里用同一组行索引。
"""

import mmap
import os
import xml.etree.ElementTree as ET

import numpy as np

from paths import IR8B as IR_DIR
XML = os.path.join(IR_DIR, "openvino_model.xml")
BIN = os.path.join(IR_DIR, "openvino_model.bin")
rng = np.random.default_rng(0)


def parse(p):
    root = ET.parse(p).getroot()
    out = {}
    for layer in root.iter("layer"):
        if layer.get("type") != "Const":
            continue
        d = layer.find("data")
        if d is None:
            continue
        k = [x for x in (d.get("shape") or "").split(",") if x.strip()]
        out[layer.get("name")] = {
            "shape": tuple(int(x) for x in k) if k else (1,),
            "et": d.get("element_type"),
            "offset": int(d.get("offset")),
            "size": int(d.get("size")),
        }
    return out


def unpack_i4(mm, c):
    r, cc = c["shape"]
    n = r * cc
    raw = np.frombuffer(mm, np.uint8, n // 2, c["offset"])
    p = np.empty(n, np.int8)
    p[0::2] = raw & 0x0F
    p[1::2] = raw >> 4
    return p.reshape(r, cc)


def main():
    consts = parse(XML)
    fh = open(BIN, "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)

    for nm in ("self.model.layers.18.mlp.up_proj.weight",
               "self.model.layers.18.mlp.down_proj.weight",
               "self.model.layers.0.self_attn.q_proj.weight"):
        W = unpack_i4(mm, consts[nm])
        sc = consts[nm + "/scale"]
        scale = np.frombuffer(mm, np.float16, int(np.prod(sc["shape"])),
                              sc["offset"]).reshape(-1, 1).astype(np.float32)

        idx = rng.choice(W.shape[0], 512, replace=False)
        sub = W[idx].astype(np.float32)
        s = scale[idx]                      # ← 同一组行

        X = (sub - 8.0) * s                 # 对称 int4: idx-8 ∈ [-8, 7]
        sig = X.std(1)
        mx = np.abs(X).max(1)
        step = s[:, 0]

        print("=" * 72)
        print(nm)
        print(f"  行数 {W.shape[0]}  scale 中位 {np.median(scale):.6f}")
        print(f"  每行 σ 中位            = {np.median(sig):.6f}")
        print(f"  每行 max|w| / σ 中位   = {np.median(mx/np.maximum(sig,1e-9)):>6.3f}"
              f"   （高斯理论 4.08）")
        print(f"  step / σ 中位          = {np.median(step/np.maximum(sig,1e-9)):>6.4f}"
              f"   （最优应为 0.3375）")
        print(f"  实际 max 撞到 ±8 的行比例 = "
              f"{100*np.mean((sub.max(1) == 15) | (sub.min(1) == 0)):.1f}%")

        # 量化 MSE 估算：step²/12（不含裁剪误差）
        mse_cur = np.median(step) ** 2 / 12
        # 最优：clip = 2.73σ, 16 电平 → step* = 2*2.73σ/16
        step_opt = 2 * 2.73 * np.median(sig) / 16
        mse_opt = step_opt ** 2 / 12
        print(f"  当前 MSE ≈ {mse_cur:.4e}  最优裁剪 MSE ≈ {mse_opt:.4e}")
        print(f"  → 只换裁剪点可回收 {10*np.log10(mse_cur/mse_opt):+.2f} dB")
        # 等效多出的比特
        print(f"  → 等效于多 {0.5*np.log2(mse_cur/mse_opt):+.3f} bit")
        print()

    # ---------------- 分布形状：真实权重是不是高斯 ----------------
    print("=" * 72)
    print("真实权重的分布形状（决定最优量化器）")
    print("=" * 72)
    nm = "self.model.layers.18.mlp.up_proj.weight"
    W = unpack_i4(mm, consts[nm])
    sc = consts[nm + "/scale"]
    scale = np.frombuffer(mm, np.float16, int(np.prod(sc["shape"])),
                          sc["offset"]).reshape(-1, 1).astype(np.float32)
    idx = rng.choice(W.shape[0], 256, replace=False)
    X = ((W[idx].astype(np.float32) - 8.0) * scale[idx]).ravel()
    X = X / X.std()
    from math import sqrt, exp, pi
    kurt = float((X ** 4).mean())
    print(f"  归一化后 kurtosis = {kurt:.3f}   （高斯 = 3.0）")
    for q, name in ((0.999, "99.9%"), (0.99, "99%"), (0.95, "95%")):
        print(f"  {name:>5} 分位 / σ = {np.quantile(np.abs(X), q):.3f}")
    print(f"  max / σ = {np.abs(X).max():.3f}")


if __name__ == "__main__":
    main()
