"""
qec_block.py — QEC 格式的表粒度：per-tensor vs per-block。

现在的 QEC v1 是**每张量一张 Huffman 表**（252 张表，7.9 KiB 索引）。
但 i4 权重的 σ 是 per-row 的（每行一个 scale），所以行与行之间分布不同，
把整张量表混在一起会抬高熵。

本脚本扫一遍块大小（16/64/256/1024 行一个块），量：
    净体积 = 码流 + 表开销
和数据：i4 张量 6.946B 参数，u8 张量单独算。
"""

import mmap
import os
import xml.etree.ElementTree as ET

import numpy as np

from paths import IR8B as IR
BINS = 16


def load_index():
    root = ET.parse(os.path.join(IR, "openvino_model.xml")).getroot()
    C = {}
    for layer in root.iter("layer"):
        if layer.get("type") != "Const":
            continue
        d = layer.find("data")
        if d is None:
            continue
        nm = layer.get("name") or ""
        if not nm.endswith(".weight"):
            continue
        k = [x for x in (d.get("shape") or "").split(",") if x.strip()]
        C[nm] = dict(shape=tuple(int(x) for x in k), et=d.get("element_type"),
                     off=int(d.get("offset")), size=int(d.get("size")))
    return C


def ent(cnt):
    p = cnt.astype(np.float64)
    s = p.sum()
    if s == 0:
        return 0.0
    p = p[p > 0] / s
    return float(-(p * np.log2(p)).sum())


def main():
    C = load_index()
    fh = open(os.path.join(IR, "openvino_model.bin"), "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)

    i4 = {k: v for k, v in C.items() if v["et"] == "i4"}
    print(f"i4 张量 {len(i4)} 个，{sum(np.prod(v['shape']) for v in i4.values())/1e9:.3f}B 参数")
    BLOCKS = [None, 1024, 256, 64, 16]
    tot = {b: 0.0 for b in BLOCKS}       # 码流 bits
    tab = {b: 0.0 for b in BLOCKS}       # 表字节
    nbar = 0

    for nm, v in i4.items():
        r, c = v["shape"]
        n = r * c
        raw = np.frombuffer(mm, np.uint8, n // 2, v["off"])
        p = np.empty(n, np.uint8)
        p[0::2] = raw & 0x0F
        p[1::2] = raw >> 4
        W = p.reshape(r, c)
        nbar += n
        cnt_all = np.bincount(W.ravel(), minlength=BINS)
        for b in BLOCKS:
            if b is None:
                tot[b] += ent(cnt_all) * n
                tab[b] += 0
            else:
                hsum = 0
                nb = 0
                for r0 in range(0, r, b):
                    blk = W[r0:r0 + b]
                    hsum += ent(np.bincount(blk.ravel(), minlength=BINS)) * blk.size
                    nb += 1
                tot[b] += hsum
                tab[b] += nb * (BINS * 2)      # 每块 16 个 16bit 计数
        del W, raw, p

    print(f"\n{'块大小':>8}{'码流 bits/权重':>16}{'码流体积':>12}{'表开销':>10}{'合计':>10}")
    for b in BLOCKS:
        bits = tot[b] / nbar
        body = nbar * bits / 8
        label = "per-tensor" if b is None else f"{b} 行"
        print(f"{label:>10}{bits:>14.4f}{body/1e9:>10.3f} GB{tab[b]/1e6:>8.1f} MB"
              f"{(body+tab[b])/1e9:>8.3f} GB")


if __name__ == "__main__":
    main()
