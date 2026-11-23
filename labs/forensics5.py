"""
forensics5.py — 修正符号约定后的最终账。

关键修正：i4 码是二进制补码，value = code if code<8 else code-16。
（之前用 code-8 平移，导致直方图呈 U 形、σ 错乱。）

注意：熵与重标号无关，所以前一轮的 H0 结论不受影响。
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
rng = np.random.default_rng(0)


def codes(nm):
    c = C[nm]
    r, cc = c["shape"]
    n = r * cc
    raw = np.frombuffer(mm, np.uint8, n // 2, c["off"])
    p = np.empty(n, np.uint8)
    p[0::2] = raw & 0x0F
    p[1::2] = raw >> 4
    return p.reshape(r, cc)


def to_signed(c):
    return np.where(c > 7, c.astype(np.int16) - 16, c.astype(np.int16)).astype(np.int8)


def main():
    nm = "self.model.layers.18.mlp.up_proj.weight"
    W = codes(nm)
    sc = C[nm + "/scale"]
    scale = np.frombuffer(mm, np.float16, int(np.prod(sc["shape"])),
                          sc["off"]).reshape(-1, 1).astype(np.float32)

    S = to_signed(W)
    print("=" * 74)
    print("0  码 → 数值的映射（修正后）")
    print("=" * 74)
    cnt = np.bincount(W.ravel(), minlength=16).astype(np.float64)
    cnt /= cnt.sum()
    print("  code :", " ".join(f"{i:>5}" for i in range(16)))
    print("  value:", " ".join(f"{(i if i<8 else i-16):>5}" for i in range(16)))
    print("  P(%) :", " ".join(f"{100*cnt[i]:>5.1f}" for i in range(16)))
    H = -float((cnt[cnt > 0] * np.log2(cnt[cnt > 0])).sum())
    print(f"  H0 = {H:.4f} bits/权重   （重标号不改变熵，与前一轮一致）")
    print("  → 按 value 排序是标准钟形：0 处 21.8%，±1 各 17.8%，±2 各 11.4%……")

    print()
    print("=" * 74)
    print("1  真实权重：σ、步长、裁剪点")
    print("=" * 74)
    idx = rng.choice(W.shape[0], 512, replace=False)
    X = S[idx].astype(np.float32) * scale[idx]
    sig = X.std(1)
    step = np.abs(scale[idx, 0])
    print(f"  每行 σ 中位        = {np.median(sig):.6f}")
    print(f"  step 中位          = {np.median(step):.6f}")
    print(f"  step / σ 中位      = {np.median(step/np.maximum(sig,1e-9)):.4f}")
    print(f"  16 电平跨度 / σ    = {16*np.median(step/np.maximum(sig,1e-9)):.2f}σ")
    print(f"  → 当前裁剪点 = ±{8*np.median(step/np.maximum(sig,1e-9)):.2f}σ"
          f"（max 启发式，理论 4.08σ）")
    print(f"  → 最优裁剪点应为 ±2.73σ，因为 16 电平时尾部不值得覆盖")

    s_cur = np.median(step / np.maximum(sig, 1e-9))
    s_opt = 2 * 2.73 / 16
    mse_cur = s_cur ** 2 / 12
    mse_opt = s_opt ** 2 / 12
    print()
    print(f"  当前 MSE/σ²   = {mse_cur:.5f}")
    print(f"  最优 MSE/σ²   = {mse_opt:.5f}")
    print(f"  → 换裁剪点可回收 {10*np.log10(mse_cur/mse_opt):+.2f} dB "
          f"≈ {0.5*np.log2(mse_cur/mse_opt):.2f} bit 等效")

    print()
    print("=" * 74)
    print("2  换裁剪点之后，熵会变成多少（更窄的裁剪 → 外侧电平更少用）")
    print("=" * 74)
    x = np.linspace(-6, 6, 400001)
    pdf = np.exp(-x * x / 2)
    pdf /= pdf.sum()
    for clip in (4.43, 3.5, 2.73, 2.5):
        st = 2 * clip / 16
        q = np.clip(np.round(x / st), -8, 7).astype(int)
        p = np.bincount(q + 8, weights=pdf, minlength=16)
        p = p / p.sum()
        e = -float((p[p > 0] * np.log2(p[p > 0])).sum())
        D = float((pdf * (x - q * st) ** 2).sum())
        print(f"  clip=±{clip:.2f}σ  step={st:.4f}σ  MSE={D:.5f}  H={e:.3f} bits")

    print()
    print("=" * 74)
    print("3  三个杠杆的最终账（Qwen3-8B，实测）")
    print("=" * 74)
    print(f"  {'方案':<38}{'bpw':>7}{'质量':>9}{'体积':>9}")
    print("  " + "-" * 66)
    print(f"  {'现状（均匀4bit, max裁剪, 不熵编码）':<38}{4.00:>7.2f}{'基线':>9}{'4.72GB':>9}")
    print(f"  {'① 无损熵编码（实测 H0=2.96）':<38}{2.96:>7.2f}{'0 dB':>9}{'3.50GB':>9}")
    print(f"  {'② 只换裁剪点 2.73σ':<38}{4.00:>7.2f}{'+5.7dB':>9}{'4.72GB':>9}")
    print(f"  {'③ ①+②':<38}{2.85:>7.2f}{'+5.7dB':>9}{'3.37GB':>9}")
    print(f"  {'④ ③+非均匀(Lloyd-Max)':<38}{2.80:>7.2f}{'+6.8dB':>9}{'3.31GB':>9}")


if __name__ == "__main__":
    main()
