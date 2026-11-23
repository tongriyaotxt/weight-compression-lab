"""
rans.py — 静态 rANS 编码器/解码器（12-bit 频率表，16 符号）。

为什么需要它：上一轮 QEC v1 用的是 canonical Huffman，它在 4 bpw 区间只贵 1.8%，
但在 sub-1.5 bpw 区间有 1 bit/符号的硬地板（三元解 H=1.31 会被编成 1.51，
稀疏二元解 H=0.43 会被编成 1.11，体积 2.5×）。所以低码率必须换 rANS。

实现：标准 32-bit rANS，8-bit 字节重归一化。
    encode:  while x >= x_max: out(x&0xFF); x >>= 8
             x = (x//f)<<12 + x%f + cum
    decode:  slot = x & (2^12-1); sym = LUT[slot]
             x = f*(x>>12) + slot - cum;  while x < L: x = x<<8 | next
"""

import numpy as np

PREC = 12
TOT = 1 << PREC
MASK = TOT - 1
L = 1 << 23


def build_table(p, nsym=16):
    """概率 -> (freq, cum) 整数表，保证总和 = TOT 且每符号 >= 1（被用到的）。"""
    p = np.asarray(p, dtype=np.float64)
    p = p / p.sum()
    used = p > 0
    f = np.zeros(nsym, np.int64)
    f[used] = np.maximum(1, np.floor(p[used] * TOT).astype(np.int64))
    # 修正总和
    while f.sum() > TOT:
        i = int(np.argmax(np.where(used, f - p * TOT, -1)))
        f[i] -= 1
    while f.sum() < TOT:
        i = int(np.argmax(np.where(used, p * TOT - f, -1)))
        f[i] += 1
    cum = np.concatenate([[0], np.cumsum(f)[:-1]])
    return f, cum


def make_lut(f, cum, nsym=16):
    lut_s = np.zeros(TOT, np.int64)
    lut_f = np.zeros(TOT, np.int64)
    lut_c = np.zeros(TOT, np.int64)
    for s in range(nsym):
        if f[s] <= 0:
            continue
        a, b = cum[s], cum[s] + f[s]
        lut_s[a:b] = s
        lut_f[a:b] = f[s]
        lut_c[a:b] = cum[s]
    return lut_s, lut_f, lut_c


def encode(syms, f, cum):
    out = bytearray()
    x = L
    fl = f.tolist()
    cl = cum.tolist()
    for s in syms:
        fs = fl[s]
        x_max = ((L >> PREC) << 8) * fs
        while x >= x_max:
            out.append(x & 0xFF)
            x >>= 8
        x = ((x // fs) << PREC) + (x % fs) + cl[s]
    return bytes(out), x


def encode_chunk(syms, f, cum, state=None):
    """从给定状态继续编码一个块，返回 (bytes, state)。大张量分块用。"""
    out = bytearray()
    x = L if state is None else state
    fl = f.tolist()
    cl = cum.tolist()
    for s in syms:
        fs = fl[s]
        x_max = ((L >> PREC) << 8) * fs
        while x >= x_max:
            out.append(x & 0xFF)
            x >>= 8
        x = ((x // fs) << PREC) + (x % fs) + cl[s]
    return bytes(out), x


def decode(data, state, f, cum, n):
    """rANS 是后进先出：按符号**倒序**解，字节也从尾部倒着取。"""
    lut_s, lut_f, lut_c = make_lut(f, cum)
    out = np.empty(n, np.int64)
    x = state
    i = len(data) - 1
    for k in range(n - 1, -1, -1):
        slot = x & MASK
        s = lut_s[slot]
        out[k] = s
        x = lut_f[slot] * (x >> PREC) + slot - lut_c[slot]
        while x < L and i >= 0:
            x = (x << 8) | data[i]
            i -= 1
    return out


def huffman_len(p):
    import heapq
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


def selftest():
    rng = np.random.default_rng(0)
    for name, p in (
        ("三元 (0.65/0.175/0.175)", np.array([0.65, 0.175, 0.175])),
        ("稀疏二元 (0.93/0.035/0.035)", np.array([0.93, 0.035, 0.035])),
        ("近似均匀 15 符号", np.array([0.218, 0.178, 0.114, 0.059, 0.026, 0.009, 0.003, 0.001,
                                   0.0, 0.001, 0.003, 0.009, 0.026, 0.059, 0.114, 0.178])),
    ):
        q = np.zeros(16)
        q[:len(p)] = p
        q /= q.sum()
        n = 200_000
        syms = rng.choice(16, size=n, p=q).tolist()
        f, cum = build_table(q)
        blob, state = encode(syms, f, cum)
        back = decode(blob, state, f, cum, n)
        ok = np.array_equal(np.array(syms), back)
        H = float(-(q[q > 0] * np.log2(q[q > 0])).sum())
        bits = len(blob) * 8 / n
        print(f"  {name:<28} H={H:5.3f}  rANS={bits:5.3f} (+{(bits/H-1)*100:4.1f}%)"
              f"  huff={huffman_len(q):5.3f} (+{(huffman_len(q)/H-1)*100:5.0f}%)  "
              f"roundtrip={'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    print("rANS 自检（200k 符号，12-bit 表）")
    selftest()
