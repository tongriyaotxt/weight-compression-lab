"""
vocab_probe.py — 词表侧（embed / lm_head）到底还能压多少。

背景：上一轮把 i4 权重压到 3.73 GB 之后，u8 的 embed + lm_head（1.245B 参数、
1187 MiB 名义、H0=7.19）变成了体积主项。而这两个张量是 u8 + **per-row 零点 + per-row scale**：

    w[i,j] = (code[i,j] - zero_point[i]) * scale[i]

也就是说：文件里存的是 code，但真正被重建的是 (code - zero_point)。
上一轮的 QEC 格式对 code 直接做熵编码，把 151936 个不同零点的偏移
全部混进同一个直方图里 —— 那 7.19 bit 里有多少是"零点散布"造成的浪费？

本脚本测四件事：
  1  H(code) vs H(code - zp)            ← 免费的那一块
  2  行内差分 / 高低 nibble 拆分
  3  每行一张表 vs 每张量一张表
  4  embed 的主子空间能否承载 lm_head（共享基）
"""

import mmap
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

from paths import IR8B as IR
CHUNK = 8192


def load_index():
    root = ET.parse(os.path.join(IR, "openvino_model.xml")).getroot()
    C = {}
    for layer in root.iter("layer"):
        if layer.get("type") != "Const":
            continue
        d = layer.find("data")
        if d is None:
            continue
        k = [x for x in (d.get("shape") or "").split(",") if x.strip()]
        C[layer.get("name")] = dict(
            shape=tuple(int(x) for x in k) if k else (1,),
            et=d.get("element_type"), off=int(d.get("offset")), size=int(d.get("size")))
    return C


def H(counts):
    p = counts[counts > 0].astype(np.float64)
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def ent_bytes(b):
    return H(np.bincount(np.frombuffer(b, np.uint8), minlength=256).astype(np.int64))


def main():
    C = load_index()
    fh = open(os.path.join(IR, "openvino_model.bin"), "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)

    for base in ("self.model.embed_tokens.weight", "self.lm_head.weight"):
        c = C[base]
        R, CC = c["shape"]
        zp = np.frombuffer(mm, np.uint8, R, C[base + "/zero_point"]["off"])
        sc = np.frombuffer(mm, np.float16, R, C[base + "/scale"]["off"]).astype(np.float32)
        print("=" * 78)
        print(f"{base}  {c['shape']}  u8  {c['size']/1e6:.0f} MB")
        print("=" * 78)
        print(f"  zero_point: min {zp.min()} max {zp.max()} mean {zp.mean():.1f} std {zp.std():.1f}")
        print(f"  scale     : median {np.median(sc):.6f}  max/min {sc.max()/max(sc.min(),1e-9):.1f}x")

        cnt_raw = np.zeros(256, np.int64)
        cnt_sub = np.zeros(256, np.int64)
        cnt_dif = np.zeros(256, np.int64)
        cnt_lo = np.zeros(16, np.int64)
        cnt_hi = np.zeros(16, np.int64)
        rowH_sub = 0.0
        nrows = 0
        for r0 in range(0, R, CHUNK):
            r1 = min(R, r0 + CHUNK)
            blk = np.frombuffer(mm, np.uint8, (r1 - r0) * CC,
                                c["off"] + r0 * CC).reshape(r1 - r0, CC)
            cnt_raw += np.bincount(blk.ravel(), minlength=256)
            sub = (blk.astype(np.int16) - zp[r0:r1, None]).astype(np.uint8)
            cnt_sub += np.bincount(sub.ravel(), minlength=256)
            cnt_lo += np.bincount((sub & 0x0F).ravel(), minlength=16)
            cnt_hi += np.bincount((sub >> 4).ravel(), minlength=16)
            d = np.diff(sub.astype(np.int16), axis=1).astype(np.uint8).ravel()
            cnt_dif += np.bincount(d, minlength=256)
            for i in range(r1 - r0):
                rowH_sub += H(np.bincount(sub[i], minlength=256))
            nrows += r1 - r0

        n = R * CC
        print(f"\n  {'方案':<34}{'bits/符号':>11}{'体积':>12}{'vs 现状':>10}")
        print("  " + "-" * 66)
        h_raw = H(cnt_raw)
        h_sub = H(cnt_sub)
        h_dif = H(cnt_dif)
        h_nib = (H(cnt_lo) + H(cnt_hi)) / 2 * 1.0
        base_mb = c["size"] / 1e6
        for name, h in (("code（现状，直接熵编码）", h_raw),
                        ("code - zero_point", h_sub),
                        ("行内一阶差分 (code-zp)", h_dif),
                        ("高低 nibble 分开（每 nibble 半字节）", h_nib * 2 / 2 * 2 / 2)):
            mb = base_mb * h / 8
            print(f"  {name:<34}{h:>11.3f}{mb:>10.0f} MB{(h/h_raw-1)*100:>9.1f}%")
        print(f"  {'每行一张表（H 的行均值）':<34}{rowH_sub/nrows:>11.3f}"
              f"{base_mb*(rowH_sub/nrows)/8:>10.0f} MB{(rowH_sub/nrows/h_raw-1)*100:>9.1f}%")

        # 主子空间：embed 与 lm_head 是否共享低秩基
        rng = np.random.default_rng(0)
        idx = rng.choice(R, 8192, replace=False)
        idx.sort()
        A = np.empty((len(idx), CC), np.float32)
        for j, i in enumerate(idx):
            blk = np.frombuffer(mm, np.uint8, CC, c["off"] + int(i) * CC)
            A[j] = (blk.astype(np.float32) - zp[i]) * sc[i]
        A -= A.mean(0, keepdims=True)
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        e = np.cumsum(S ** 2) / (S ** 2).sum()
        print("\n  主子空间（8192 行抽样，行中心化后）")
        for k in (16, 32, 64, 128, 256):
            print(f"    rank-{k:<4} 能量捕获 {e[k-1]*100:5.1f}%")
        np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dz",
                             f"V_{base.split('.')[-2]}.npy"), Vt[:256].astype(np.float32))


if __name__ == "__main__":
    main()
