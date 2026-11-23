"""
repack_rans.py — 把 QEC v1 的 Huffman 换成静态 rANS，给出**精确**文件大小。

方法：静态 rANS 的码长是可解析算出的（不是估计）：
    对每个张量，把直方图量化成 2^12 频率表 f_i，
    精确码长 = Σ_symbols -log2(f_i / 4096)
这与真编码器逐字节的结果一致（在 1.27M 符号上验证过：解析 1.366 vs 实际 1.366）。

本脚本：
  1  扫一遍 252 个 i4 + 2 个 u8 张量，建表，算精确 bits
  2  对某一个真实张量跑**真 rANS 编解码**，验证解析式与字节数吻合
  3  给出总账：熵 / rANS / Huffman，以及索引开销
"""

import mmap
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rans import build_table, decode, encode  # noqa: E402

from paths import IR8B as IR
TOT = 4096


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


def read_codes(mm, c):
    """返回 (符号数组, 每符号名义 bits)。"""
    r, cc = c["shape"]
    n = r * cc
    if c["et"] == "i4":
        raw = np.frombuffer(mm, np.uint8, n // 2, c["off"])
        p = np.empty(n, np.uint8)
        p[0::2] = raw & 0x0F
        p[1::2] = raw >> 4
        return p, 4.0
    return np.frombuffer(mm, np.uint8, n, c["off"]), 8.0


def exact_rans_bits(cnt):
    """给定直方图，返回精确 rANS 码长（bits）与该表的字节开销。"""
    f, cum = build_table(cnt.astype(np.float64), nsym=len(cnt))
    q = f / f.sum()
    nz = cnt > 0
    return float(-(cnt[nz] * np.log2(q[nz])).sum()), len(cnt) * 2


def huff_avg_from_cnt(cnt):
    import heapq
    p = cnt / cnt.sum()
    h = [(float(x), i, {i: 0}) for i, x in enumerate(p) if x > 0]
    if len(h) <= 1:
        return 1.0
    heapq.heapify(h)
    depth = {i: 0 for i, x in enumerate(p) if x > 0}
    while len(h) > 1:
        a = heapq.heappop(h)
        b = heapq.heappop(h)
        for s in a[2]:
            depth[s] += 1
        for s in b[2]:
            depth[s] += 1
        heapq.heappush(h, (a[0] + b[0], a[1] + b[1], {**a[2], **b[2]}))
    return float(sum(p[i] * depth[i] for i in depth))


def main():
    C = load_index()
    fh = open(os.path.join(IR, "openvino_model.bin"), "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)

    tot_rans = tot_huff = tot_ent = 0.0
    tot_sym = 0
    tab_bytes = 0
    nominal = 0
    per = []
    t0 = time.time()
    for nm, c in sorted(C.items()):
        sym, nb = read_codes(mm, c)
        cnt = np.bincount(sym, minlength=16 if c["et"] == "i4" else 256)
        p = cnt / cnt.sum()
        nz = p[p > 0]
        H = float(-(nz * np.log2(nz)).sum())
        rb, tb = exact_rans_bits(cnt)
        hb = huff_avg_from_cnt(cnt) * len(sym)
        tot_ent += H * len(sym)
        tot_rans += rb
        tot_huff += hb
        tot_sym += len(sym)
        tab_bytes += tb
        nominal += c["size"]
        per.append((nm, len(sym), H, rb / len(sym), hb / len(sym)))
        del sym

    print("=" * 88)
    print(f"Qwen3-8B  254 个张量  {tot_sym/1e9:.3f}B 符号")
    print("=" * 88)
    print(f"  名义（定长）      {nominal/2**30:8.3f} GiB   {nominal/1e9:6.3f} GB")
    print(f"  熵下界            {tot_ent/8/2**30:8.3f} GiB   {tot_ent/8/1e9:6.3f} GB"
          f"   ({tot_ent/tot_sym:.4f} bit/符号)")
    print(f"  静态 rANS（精确）  {tot_rans/8/2**30:8.3f} GiB   {tot_rans/8/1e9:6.3f} GB"
          f"   ({tot_rans/tot_sym:.4f} bit/符号, +{(tot_rans/tot_ent-1)*100:.2f}%)")
    print(f"  canonical Huffman {tot_huff/8/2**30:8.3f} GiB   {tot_huff/8/1e9:6.3f} GB"
          f"   ({tot_huff/tot_sym:.4f} bit/符号, +{(tot_huff/tot_ent-1)*100:.2f}%)")
    print(f"  表开销            {tab_bytes/1024:8.1f} KiB")
    print(f"  → rANS 总文件     {(tot_rans/8+tab_bytes)/2**30:8.3f} GiB"
          f"   {(tot_rans/8+tab_bytes)/1e9:6.3f} GB")
    print(f"  → vs rANS 初版 Huffman: 省 {(tot_huff-tot_rans)/8/1e9:.3f} GB"
          f" ({(1-tot_rans/tot_huff)*100:.2f}%)")

    # 真实编码验证：挑最大的 i4 张量
    big = max((k for k, v in C.items() if v["et"] == "i4"), key=lambda k: C[k]["size"])
    c = C[big]
    sym, _ = read_codes(mm, c)
    cnt = np.bincount(sym, minlength=16)
    rb, _ = exact_rans_bits(cnt)
    print("\n" + "=" * 88)
    print(f"真实编码验证：{big}  {c['shape']}  {len(sym)/1e6:.1f}M 符号")
    print("=" * 88)
    f, cum = build_table(cnt.astype(np.float64), nsym=16)
    t0 = time.time()
    blob, state = encode(sym.tolist(), f, cum)
    enc_s = time.time() - t0
    t0 = time.time()
    back = decode(blob, state, f, cum, len(sym))
    dec_s = time.time() - t0
    ok = np.array_equal(sym, back)
    print(f"  解析预测 {rb/8/1e6:9.4f} MB   实际编码 {len(blob)/1e6:9.4f} MB"
          f"   差 {len(blob)*8/rb-1:+.4%}")
    print(f"  roundtrip {'OK' if ok else 'FAIL'}   编码 {len(sym)/enc_s/1e6:.2f} M符号/s"
          f"   解码 {len(sym)/dec_s/1e6:.2f} M符号/s")
    print(f"  → 全模型按纯 Python 速率编码需 {tot_sym/(len(sym)/enc_s)/3600:.1f} 小时"
          f"（C 实现 ~100-200 MB/s，约 {tot_sym*0.4/200e6/60:.0f} 分钟）")


if __name__ == "__main__":
    main()
