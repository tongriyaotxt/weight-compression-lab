"""
codec.py — 真刀真枪：在真实权重上跑 Huffman 和静态 rANS。

验证三件事：
  1. H0=3.04 能不能被真正的编码器达到（表量化误差有多大）
  2. Huffman 比 rANS 差多少（速度/体积的权衡）
  3. 我们离 R(D) 界还有多远 —— 剩下的空间只可能来自矢量量化
"""

import mmap
import os
import time
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

SCALE_BITS = 12
M = 1 << SCALE_BITS
RANS_L = 1 << 23


def get_codes(nm, limit=None):
    c = C[nm]
    r, cc = c["shape"]
    n = r * cc
    if limit:
        n = min(n, limit)
    raw = np.frombuffer(mm, np.uint8, n // 2, c["off"])
    p = np.empty(n, np.uint8)
    p[0::2] = raw & 0x0F
    p[1::2] = raw >> 4
    return p


# ---------------------------------------------------------------- Huffman
def huffman(sym):
    cnt = np.bincount(sym, minlength=16).astype(np.float64)
    prob = cnt / cnt.sum()
    H = -float((prob[prob > 0] * np.log2(prob[prob > 0])).sum())

    # 构造 Huffman 树
    import heapq
    h = [(cnt[i], i, (i,)) for i in range(16) if cnt[i] > 0]
    heapq.heapify(h)
    codes = {}
    if len(h) == 1:
        codes[h[0][2][0]] = 1
    while len(h) > 1:
        a = heapq.heappop(h)
        b = heapq.heappop(h)
        for s in a[2]:
            codes[s] = codes.get(s, 0)
        for s in b[2]:
            codes[s] = codes.get(s, 0)
        heapq.heappush(h, (a[0] + b[0], -min(a[1], b[1]), a[2] + b[2]))
    # 按树深度重新分配（自底向上累加长度）
    depth = {s: 0 for s in range(16) if cnt[s] > 0}
    stack = [(h[0][2], 0)]
    while stack:
        grp, d = stack.pop()
        if len(grp) == 1:
            depth[grp[0]] = max(d, 1)
        else:
            half = len(grp) // 2
            stack.append((grp[:half], d + 1))
            stack.append((grp[half:], d + 1))
    avg = float(sum(prob[s] * depth[s] for s in depth))
    return H, avg, depth


# ---------------------------------------------------------------- rANS
def build_model(sym):
    cnt = np.bincount(sym, minlength=16).astype(np.float64)
    freq = np.maximum(np.round(cnt / cnt.sum() * M), 1).astype(np.int64)
    # 调整到恰好 M
    while freq.sum() > M:
        j = int(np.argmax(freq - (freq > 1) * 1e9 if (freq > 1).any() else freq))
        freq[j] -= 1
    while freq.sum() < M:
        freq[int(np.argmax(freq))] += 1
    cum = np.concatenate([[0], np.cumsum(freq)[:-1]])
    return freq, cum


def rans_encode(sym, freq, cum):
    out = bytearray()
    state = RANS_L
    fl = freq.tolist()
    cl = cum.tolist()
    for s in reversed(sym.tolist()):
        f = fl[s]
        x_max = ((RANS_L >> SCALE_BITS) << 8) * f
        while state >= x_max:
            out.append(state & 0xFF)
            state >>= 8
        state = ((state // f) << SCALE_BITS) + (state % f) + cl[s]
    out.extend(state.to_bytes(4, "little"))
    return bytes(out)


def rans_decode(blob, freq, cum, n):
    tab = np.zeros(M, dtype=np.uint8)
    for s in range(16):
        tab[cum[s]:cum[s] + freq[s]] = s
    fl = freq.tolist()
    cl = cum.tolist()
    state = int.from_bytes(blob[-4:], "little")
    pos = len(blob) - 5
    out = np.empty(n, dtype=np.uint8)
    for i in range(n):
        slot = state & (M - 1)
        s = int(tab[slot])
        state = fl[s] * (state >> SCALE_BITS) + slot - cl[s]
        while state < RANS_L:
            state = (state << 8) | blob[pos]
            pos -= 1
        out[i] = s
    return out


def main():
    nm = "self.model.layers.18.mlp.up_proj.weight"
    N = 400_000
    sym = get_codes(nm, N)
    print(f"样本: {nm}  取前 {len(sym)} 个符号\n")

    # ---------------- 1. 熵 ----------------
    cnt = np.bincount(sym, minlength=16).astype(np.float64)
    p = cnt / cnt.sum()
    H = -float((p[p > 0] * np.log2(p[p > 0])).sum())
    print("=" * 70)
    print("1  熵")
    print("=" * 70)
    print(f"  H0 = {H:.4f} bits/符号   名义 4.0   → 省 {100*(1-H/4):.1f}%")

    # ---------------- 2. Huffman ----------------
    print()
    print("=" * 70)
    print("2  Canonical Huffman")
    print("=" * 70)
    H2, avg, depth = huffman(sym)
    print(f"  平均码长 = {avg:.4f} bits   （比熵多 {avg-H:.4f}）")
    print(f"  码长表   = {dict(sorted(depth.items()))}")
    print(f"  最大码长 = {max(depth.values())} bits  → 解码可用单次 "
          f"{1<<max(depth.values())} 项 LUT")
    print(f"  体积     = {avg/4*100:.1f}% of 4bit")

    # ---------------- 3. rANS ----------------
    print()
    print("=" * 70)
    print("3  Static rANS (2^12 表, 16 符号)")
    print("=" * 70)
    freq, cum = build_model(sym)
    print(f"  频率表 = {freq.tolist()}")
    print(f"  表误差 = Σ|p̂-p| = {np.abs(freq/M - p).sum():.5f}")

    t0 = time.perf_counter()
    blob = rans_encode(sym, freq, cum)
    t_enc = time.perf_counter() - t0
    rate = len(blob) * 8 / len(sym)
    print(f"  编码后   = {len(blob)} 字节 / {len(sym)} 符号")
    print(f"  实际码率 = {rate:.4f} bits/符号   （比熵多 {rate-H:+.4f}）")
    print(f"  编码吞吐 = {len(sym)/t_enc/1e6:.2f} M符号/s  (纯 Python，仅供参考)")

    t0 = time.perf_counter()
    dec = rans_decode(blob, freq, cum, len(sym))
    t_dec = time.perf_counter() - t0
    ok = np.array_equal(dec, sym.astype(np.uint8))
    print(f"  解码吞吐 = {len(sym)/t_dec/1e6:.2f} M符号/s  (纯 Python)")
    print(f"  roundtrip: {'✅ 完全一致' if ok else '❌ 不一致'}")

    # ---------------- 4. 离 R(D) 有多远 ----------------
    print()
    print("=" * 70)
    print("4  离信息论极限还有多远")
    print("=" * 70)
    step_over_sigma = 0.4934          # 上一轮实测
    D = step_over_sigma ** 2 / 12     # 归一化 MSE/σ²
    R_D = 0.5 * np.log2(1 / D)
    print(f"  实测 step/σ          = {step_over_sigma:.4f}")
    print(f"  量化失真 D/σ²        = {D:.5f}")
    print(f"  R(D) 高斯下界        = {R_D:.4f} bits  ← 任何编码器的绝对地板")
    print(f"  我们达到             = {H:.4f} bits")
    print(f"  差距                 = {H-R_D:+.4f} bits  ({100*(H-R_D)/R_D:.1f}%)")
    print()
    print(f"  标量最优（ECSQ）     = R(D) + 0.255 = {R_D+0.255:.4f} bits"
          f"  ← 我们 {'已经达到' if H <= R_D+0.26 else '还差一点'}")
    print(f"  → 这 0.25 bit 的差距是空间的 1.53 dB 格填充损失，")
    print(f"    只有矢量量化（VQ/格/AQLM）能拿回来。")

    # ---------------- 5. 硬件判断 ----------------
    print()
    print("=" * 70)
    print("5  在你的卡上，熵解码能不能跟上带宽")
    print("=" * 70)
    BW = 670e9
    ops_per_sym = 9          # rANS: LUT+乘+移位+加+掩码+renorm
    gpu_ops = 1.2e13         # 5070Ti: ~5888 core × ~2 GHz
    need = BW * 8 / rate
    can = gpu_ops / ops_per_sym
    print(f"  3.04 bpw 下，吃满 670 GB/s 需要解 {need/1e12:.2f} T符号/s")
    print(f"  GPU 大约能做             {can/1e12:.2f} T符号/s（{ops_per_sym} op/符号）")
    print(f"  → 解码 {'是瓶颈，约能跑到 ' + str(round(100*can/need)) + '% 带宽' if can < need else '不是瓶颈'}")


if __name__ == "__main__":
    main()
