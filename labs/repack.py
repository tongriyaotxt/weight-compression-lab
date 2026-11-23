"""
repack.py — 把方法论变成格式：QEC (Quantize + Entropy Code)。

产出：
  1. 每个张量的 Huffman 码长表（16 字节/张量）
  2. 精确的最终文件大小
  3. 一个真的 bit-writer/bit-reader，在样本上做 roundtrip 验证
  4. 写出 model.qec.json 索引

格式设计（这就是本轮的方法论结论）：
  - 量化器只负责塑造**边际**（电平数、裁剪点）
  - 熵编码器只负责**读边际**（静态 Huffman，单次 LUT 解码）
  - 稀疏性不再是格式问题，它只是 P(0) 变大，自动被编码器吃掉
"""

import json
import mmap
import os
import xml.etree.ElementTree as ET

import numpy as np

from paths import IR8B as IR
from paths import HERE as _H
OUT = os.path.join(_H, "qec")
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
                                off=int(d.get("offset", 0)),
                                size=int(d.get("size")))
fh = open(os.path.join(IR, "openvino_model.bin"), "rb")
mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)


# ---------------------------------------------------------------- Huffman
def huff_lengths(cnt):
    import heapq
    h = [(cnt[i], i) for i in range(len(cnt)) if cnt[i] > 0]
    heapq.heapify(h)
    parent = {}
    nid = len(cnt)
    while len(h) > 1:
        a = heapq.heappop(h)
        b = heapq.heappop(h)
        parent[a[1]] = nid
        parent[b[1]] = nid
        heapq.heappush(h, (a[0] + b[0], nid))
        nid += 1
    d = {}
    for s in range(len(cnt)):
        if cnt[s] == 0:
            continue
        node, dep = s, 0
        while node in parent:
            node = parent[node]
            dep += 1
        d[s] = max(dep, 1)
    return d


def canonical(d):
    """返回 (code, length) 字典。"""
    items = sorted(d.items(), key=lambda kv: (kv[1], kv[0]))
    out = {}
    code = 0
    prev = items[0][1]
    for s, L in items:
        code <<= (L - prev)
        prev = L
        out[s] = (code, L)
        code += 1
    return out


class BitWriter:
    def __init__(self):
        self.acc = 0
        self.n = 0
        self.buf = bytearray()

    def write(self, code, length):
        self.acc = (self.acc << length) | code
        self.n += length
        while self.n >= 8:
            self.n -= 8
            self.buf.append((self.acc >> self.n) & 0xFF)
        self.acc &= (1 << self.n) - 1

    def finish(self):
        if self.n:
            self.buf.append((self.acc << (8 - self.n)) & 0xFF)
            self.n = 0
        return bytes(self.buf)


class BitReader:
    def __init__(self, data):
        self.d = data
        self.pos = 0
        self.acc = 0
        self.n = 0

    def _fill(self, k):
        while self.n < k and self.pos < len(self.d):
            self.acc = (self.acc << 8) | self.d[self.pos]
            self.pos += 1
            self.n += 8
        while self.n < k:                      # 末尾补零
            self.acc <<= 8
            self.n += 8

    def peek(self, k):
        self._fill(k)
        return (self.acc >> (self.n - k)) & ((1 << k) - 1)

    def skip(self, k):
        self.n -= k
        self.acc &= (1 << self.n) - 1


def decode_table(codes, maxlen):
    """反向表：读 maxlen 位 → (symbol, length)"""
    n = 1 << maxlen
    tab = np.full((n, 2), -1, dtype=np.int16)
    for s, (c, L) in codes.items():
        shift = maxlen - L
        base = c << shift
        for k in range(1 << shift):
            tab[base + k] = (s, L)
    return tab


def main():
    os.makedirs(OUT, exist_ok=True)
    entries = []
    tot_bytes = 0
    tot_symbols = 0
    tot_bits = 0.0
    codec_stats = []

    for nm, c in sorted(C.items()):
        if c["et"] not in ("i4", "u8"):
            continue
        if not nm.endswith(".weight"):
            continue
        r, cc = c["shape"]
        n = r * cc
        if c["et"] == "i4":
            raw = np.frombuffer(mm, np.uint8, n // 2, c["off"])
            sym = np.empty(n, np.uint8)
            sym[0::2] = raw & 0x0F
            sym[1::2] = raw >> 4
            nbits_sym = 4
        else:
            sym = np.frombuffer(mm, np.uint8, n, c["off"])
            nbits_sym = 8

        K = 16 if c["et"] == "i4" else 256
        cnt = np.bincount(sym, minlength=K).astype(np.int64)
        p = cnt / cnt.sum()
        pnz = p[p > 0]
        H = float(-(pnz * np.log2(pnz)).sum())
        d = huff_lengths(cnt)
        codes = canonical(d)
        avg = float(sum(p[s] * d[s] for s in d))
        nbytes = int(np.ceil(n * avg / 8))
        tot_bytes += nbytes
        tot_symbols += n
        tot_bits += n * avg
        entries.append(dict(name=nm, shape=list(c["shape"]), et=c["et"],
                            offset=c["off"], n=n, bpw=round(4 * avg / nbits_sym, 4),
                            huff=[[int(s), int(d[s])] for s in sorted(d)]))
        codec_stats.append((nm, H, avg, n))

    print("=" * 78)
    print("1  逐张量码率（部分）")
    print("=" * 78)
    print(f"  {'tensor':<48}{'H0':>7}{'huff':>7}{'bpw':>7}")
    print("  " + "-" * 69)
    for nm, H, avg, n in codec_stats[:6] + codec_stats[-3:]:
        short = nm.replace("self.model.", "")
        base = 4 if "layers" in nm else 8
        print(f"  {short:<48}{H:>7.4f}{avg:>7.4f}{4*avg/base:>7.3f}")

    print()
    print("=" * 78)
    print("2  全模型")
    print("=" * 78)
    print(f"  符号总数        : {tot_symbols/1e9:.3f} B")
    print(f"  平均码率        : {tot_bits/tot_symbols:.4f} bit/符号")
    print(f"  压缩后          : {tot_bytes/2**20:.1f} MiB  ({tot_bytes/1e9:.2f} GB)")
    print(f"  原始            : {sum(c['size'] for c in C.values() if c['et'] in ('i4','u8'))/2**20:.1f} MiB")
    print(f"  索引开销        : {len(entries)*16*2/1024:.1f} KiB (码长表)")
    print(f"  总和            : {(tot_bytes + len(entries)*16*2)/1e9:.3f} GB")

    # ---------------- roundtrip 验证 ----------------
    print()
    print("=" * 78)
    print("3  真实 bit-writer / bit-reader roundtrip")
    print("=" * 78)
    nm = "self.model.layers.18.mlp.up_proj.weight"
    c = C[nm]
    r, cc = c["shape"]
    SAMP = 300_000
    raw = np.frombuffer(mm, np.uint8, SAMP // 2, c["off"])
    sym = np.empty(SAMP, np.uint8)
    sym[0::2] = raw & 0x0F
    sym[1::2] = raw >> 4
    cnt = np.bincount(sym, minlength=16).astype(np.int64)
    d = huff_lengths(cnt)
    codes = canonical(d)
    bw = BitWriter()
    for s in sym.tolist():
        cd, L = codes[s]
        bw.write(cd, L)
    blob = bw.finish()
    maxlen = max(v[1] for v in codes.values())
    tab = decode_table(codes, maxlen)
    br = BitReader(blob)
    out = np.empty(SAMP, np.uint8)
    for i in range(SAMP):
        v = tab[br.peek(maxlen)]
        out[i] = v[0]
        br.skip(int(v[1]))
    ok = np.array_equal(out, sym)
    theory = sum(cnt[s] / cnt.sum() * d[s] for s in d)
    print(f"  样本 {SAMP} 符号 → {len(blob)} 字节 = {len(blob)*8/SAMP:.4f} bit/符号")
    print(f"  理论 avg = {theory:.4f}   maxlen = {maxlen}   LUT = {1<<maxlen} 项")
    print(f"  roundtrip: {'✅ 完全一致' if ok else '❌ 不一致'}")

    with open(os.path.join(OUT, "model.qec.json"), "w") as f:
        json.dump({"source": IR, "format": "QEC v1",
                   "symbols": int(tot_symbols),
                   "bits": int(tot_bits),
                   "bytes": int(tot_bytes),
                   "tensors": entries}, f, indent=1)
    print()
    print(f"  索引已写出: {OUT}\\model.qec.json  "
          f"({os.path.getsize(os.path.join(OUT,'model.qec.json'))/1024:.0f} KiB)")


if __name__ == "__main__":
    main()
