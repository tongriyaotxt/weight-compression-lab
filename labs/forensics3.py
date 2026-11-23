"""
forensics3.py — 第三轮：把账算到底。

1) 全模型 252 个 i4 张量的实测 H0 → 无损熵编码后 Qwen3-8B 到底多大
2) 反量化出真实权重，量出"max 裁剪"到底浪费了多少 dB
3) 三个杠杆的独立贡献：熵编码 / 最优裁剪 / 非均匀量化
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


def h0(vals, cnt):
    p = cnt / cnt.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def main():
    consts = parse(XML)
    fh = open(BIN, "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)

    i4 = {k: v for k, v in consts.items() if v["et"] == "i4"}
    u8 = {k: v for k, v in consts.items() if v["et"] == "u8"}

    # ---------------- 1) 全模型实测 ----------------
    print("=" * 78)
    print("1  全模型：无损熵编码之后有多大")
    print("=" * 78)
    tot_params = 0
    tot_nominal = 0.0     # bits
    tot_entropy = 0.0     # bits
    per_h = []
    for nm, c in sorted(i4.items()):
        W = unpack_i4(mm, c)
        sub = W[rng.choice(W.shape[0], min(384, W.shape[0]), replace=False)]
        cnt = np.bincount(sub.ravel(), minlength=16).astype(np.float64)
        e = h0(sub, cnt)
        per_h.append(e)
        n = int(np.prod(c["shape"]))
        tot_params += n
        tot_nominal += 4.0 * n
        tot_entropy += e * n

    e_u8 = []
    n_u8 = 0
    nom_u8 = 0.0
    ent_u8 = 0.0
    for nm, c in sorted(u8.items()):
        n = int(np.prod(c["shape"]))
        rows = c["shape"][0]
        raw = np.frombuffer(mm, np.uint8, n, c["offset"])
        sub = raw.reshape(c["shape"])[rng.choice(rows, min(384, rows), replace=False)].ravel()
        cnt = np.bincount(sub, minlength=256).astype(np.float64)
        e = h0(sub, cnt)
        e_u8.append(e)
        n_u8 += n
        nom_u8 += 8.0 * n
        ent_u8 += e * n

    print(f"  i4 张量: {len(i4)} 个，参数 {tot_params/1e9:.3f} B")
    print(f"    H0: mean={np.mean(per_h):.4f}  min={np.min(per_h):.4f}  max={np.max(per_h):.4f}")
    print(f"    名义 {tot_nominal/8/2**20:8.1f} MiB  →  熵 {tot_entropy/8/2**20:8.1f} MiB"
          f"   省 {100*(1-tot_entropy/tot_nominal):.1f}%")
    print(f"  u8 张量: {len(u8)} 个（embedding/lm_head），参数 {n_u8/1e9:.3f} B")
    print(f"    H0: {np.mean(e_u8):.4f}")
    print(f"    名义 {nom_u8/8/2**20:8.1f} MiB  →  熵 {ent_u8/8/2**20:8.1f} MiB"
          f"   省 {100*(1-ent_u8/nom_u8):.1f}%")

    nominal = (tot_nominal + nom_u8) / 8 / 2**20
    entropy = (tot_entropy + ent_u8) / 8 / 2**20
    print()
    print(f"  合计名义权重 : {nominal:8.1f} MiB")
    print(f"  合计熵       : {entropy:8.1f} MiB")
    print(f"  → 无损压缩后 : {entropy:8.1f} MiB  （省 {nominal-entropy:.1f} MiB，"
          f"{100*(1-entropy/nominal):.1f}%）")
    print(f"  → 等比例外推到 30B (4bit, 15 GB): {15*entropy/nominal:.1f} GB")

    # ---------------- 2) 反量化，量 max 裁剪的浪费 ----------------
    print()
    print("=" * 78)
    print("2  反量化真实权重：max 裁剪 vs 最优裁剪")
    print("=" * 78)
    nm = "self.model.layers.18.mlp.up_proj.weight"
    W = unpack_i4(mm, consts[nm]).astype(np.float32)
    sc = consts[nm + "/scale"]
    scale = np.frombuffer(mm, np.float16, int(np.prod(sc["shape"])), sc["offset"])
    scale = scale.reshape(-1, 1).astype(np.float32)
    print(f"  {nm}")
    print(f"  idx 范围 {W.min():.0f}..{W.max():.0f}   scale shape {scale.shape}"
          f"   scale 中位 {np.median(scale):.6f}")

    sub = W[rng.choice(W.shape[0], 512, replace=False)]
    s = scale[rng.choice(scale.shape[0], 512, replace=False)]
    X = (sub - 7.5) * s                      # 反量化（零点在 7.5）
    sig = X.std(1)
    mx = np.abs(X).max(1)
    print(f"  每行 σ 中位 = {np.median(sig):.6f}")
    print(f"  每行 max|w| / σ 中位 = {np.median(mx/sig):.3f}   ← 理论上高斯应约 3.3–4.1")
    print(f"  当前 step / σ 中位 = {np.median(s/np.maximum(sig,1e-9)):.4f}")

    # 当前 MSE ≈ step²/12；最优 uniform 16 电平 step* ≈ 0.3375σ
    cur = (np.median(s) ** 2) / 12
    opt = ((2 * 2.7 * np.median(sig) / 16) ** 2) / 12
    print(f"  当前量化 MSE      ≈ {cur:.3e}")
    print(f"  最优裁剪 MSE      ≈ {opt:.3e}")
    print(f"  → 仅换裁剪点，可回收 {10*np.log10(cur/opt):.2f} dB")

    # ---------------- 3) 三个杠杆 ----------------
    print()
    print("=" * 78)
    print("3  三个杠杆的独立贡献（Qwen3-8B）")
    print("=" * 78)
    rows = [
        ("现状（4bit 均匀 + max 裁剪 + 不熵编码）", 4.00, 0.0),
        ("① 加无损熵编码（静态表 rANS）", np.mean(per_h), 0.0),
        ("② 只换最优裁剪点（不动格式）", 4.00, 3.4),
        ("③ ①+② 同时做", np.mean(per_h), 3.4),
        ("④ 再加非均匀量化（Lloyd-Max/ECSQ）", np.mean(per_h), 3.4 + 1.1),
    ]
    print(f"  {'方案':<40}{'存储 bpw':>10}{'质量增益':>10}{'体积':>10}")
    print("  " + "-" * 70)
    for name, b, db in rows:
        vol = 4.72 * b / 4.0
        print(f"  {name:<40}{b:>10.2f}{db:>9.1f}dB{vol:>9.2f}GB")


if __name__ == "__main__":
    main()
