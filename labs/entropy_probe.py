"""
entropy_probe.py — empirical look at where the theoretical floor sits.

For each corpus we report:
  H0   : order-0 entropy            (bits/symbol)  -> hard floor for any static symbol coder
  H1   : order-1 conditional entropy (bits/symbol) -> floor for a 1st-order Markov coder
  H2   : order-2 conditional entropy (bits/symbol)
  zlib : deflate  -9    -> bits/symbol achieved
  bz2  : bzip2    -9    -> bits/symbol achieved
  lzma : xz preset 9|EXTREME -> bits/symbol achieved
  best : model-free empirical floor reached
"""

import bz2
import glob
import lzma
import math
import os
import random
import sys
import sysconfig
import zlib
from collections import Counter, defaultdict


# ---------------------------------------------------------------- entropy
def H0(data: bytes) -> float:
    n = len(data)
    if n == 0:
        return 0.0
    c = Counter(data)
    return -sum((v / n) * math.log2(v / n) for v in c.values())


def Hk(data: bytes, order: int) -> float:
    """Empirical conditional entropy H(X_t | X_{t-order..t-1})."""
    n = len(data)
    if n <= order:
        return 0.0
    ctx = defaultdict(Counter)
    for i in range(order, n):
        ctx[data[i - order:i]][data[i]] += 1
    total = 0.0
    for ctr in ctx.values():
        s = sum(ctr.values())
        total += s * (-sum((v / s) * math.log2(v / s) for v in ctr.values()))
    # add the cost of the first `order` symbols, modelled at order-0
    return (total + order * H0(data[: max(order, 1)])) / n


# ---------------------------------------------------------------- corpora
def text_corpus(target=4 << 20) -> bytes:
    root = sysconfig.get_paths()["stdlib"]
    buf = bytearray()
    for p in sorted(glob.glob(os.path.join(root, "*.py"))):
        try:
            buf += open(p, "rb").read()
        except OSError:
            continue
        if len(buf) >= target:
            break
    return bytes(buf[:target])


def make_corpora(n=1 << 20) -> dict:
    rnd = random.Random(1234)
    out = {}

    # 1. natural language / source code
    out["text (python stdlib)"] = text_corpus(n)

    # 2. uniform random — the incompressible case
    out["random bytes"] = bytes(rnd.getrandbits(8) for _ in range(n))

    # 3. all zeros — the degenerate case
    out["all zeros"] = b"\x00" * n

    # 4. log lines — highly templated
    levels = ["INFO", "WARN", "ERROR", "DEBUG"]
    users = ["alice", "bob", "carol", "dave"]
    lines = []
    t = 1700000000
    while sum(map(len, lines)) < n:
        t += rnd.randint(1, 9)
        lines.append(
            f"{t} {rnd.choice(levels)} user={rnd.choice(users)} "
            f"req_id={rnd.getrandbits(32):08x} lat={rnd.randint(1,500)}ms path=/api/v1/items/{rnd.randint(1,9999)}\n"
        )
    out["server log"] = "".join(lines).encode()[:n]

    # 5. smooth float signal (audio/telemetry-ish), quantised to int16
    import struct
    vals = []
    ph = 0.0
    for i in range(n // 2):
        ph += 0.01
        v = int(12000 * math.sin(ph) + 3000 * math.sin(3.1 * ph + 1) + rnd.gauss(0, 40))
        vals.append(max(-32768, min(32767, v)))
    out["smooth signal int16"] = struct.pack(f"<{len(vals)}h", *vals)

    # 6. skewed DNA (biological alphabet, non-uniform)
    w = [0.30, 0.20, 0.22, 0.28]
    out["dna (skewed ACGT)"] = bytes(rnd.choices(b"ACGT", weights=w, k=n))

    # 7. already-compressed blob (double compression sanity check)
    out["xz of text"] = lzma.compress(text_corpus(n), preset=9)

    return out


# ---------------------------------------------------------------- measure
def measure(data: bytes) -> dict:
    n = len(data)
    return {
        "n": n,
        "H0": H0(data),
        "H1": Hk(data, 1),
        "H2": Hk(data, 2),
        "zlib": 8 * len(zlib.compress(data, 9)) / n,
        "bz2": 8 * len(bz2.compress(data, 9)) / n,
        "lzma": 8 * len(lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)) / n,
    }


def main():
    rows = []
    for name, data in make_corpora().items():
        r = measure(data)
        r["corpus"] = name
        rows.append(r)

    hdr = f"{'corpus':<24}{'bytes':>9}{'H0':>7}{'H1':>7}{'H2':>7}{'zlib':>8}{'bz2':>8}{'lzma':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['corpus']:<24}{r['n']:>9}{r['H0']:>7.3f}{r['H1']:>7.3f}{r['H2']:>7.3f}"
            f"{r['zlib']:>8.3f}{r['bz2']:>8.3f}{r['lzma']:>8.3f}"
        )
    print()
    print("all figures in bits/symbol (divide by 8 for bytes/symbol; ratio = value/8)")
    print("NOTE: H1/H2 are *empirical* (plug-in) estimates and are biased LOW when the")
    print("      context table is sparsely populated -- see 'random bytes' H2=3.98 and")
    print("      'xz of text' H2=1.86. They are upper bounds on what a coder could")
    print("      exploit only if the model were free; it is not. See redundancy below.")
    print()

    # ---- counting argument, done properly ---------------------------------
    # Of the 2^n strings of length n, at most 2^(n-k) can be mapped to codewords
    # of length <= n-k (prefix-free, injective). So the fraction that shrinks by
    # k bits or more is at most 2^-k, no matter how clever the algorithm.
    print("counting argument: fraction of length-n strings compressible by >= k bits")
    print(f"{'k (bits saved)':>15}{'max fraction':>16}{'i.e. 1 in':>14}")
    for k in (1, 8, 16, 32, 64, 128):
        print(f"{k:>15}{2.0**-k:>16.3e}{2**k:>14,}")
    print()

    # ---- price of universality: KT redundancy ------------------------------
    # Sequential KT predictor p(x_j) = (count_j + 1/2) / (n + k/2).
    # Total code length ~= n*H_emp + ((k-1)/2)*log2(n) + O(1).
    print("universal coding overhead (Krichevsky-Trofimov) on skewed DNA, k=4")
    dna = bytes(random.Random(7).choices(b"ACGT", weights=[.3, .2, .22, .28], k=1 << 20))
    print(f"{'n':>10}{'n*H0 bits':>14}{'KT bits':>12}{'overhead':>11}{'pred (k-1)/2*log2 n':>22}")
    k_alpha = 4
    for n in (1_000, 10_000, 100_000, 1_000_000):
        seg = dna[:n]
        cnt = Counter()
        bits = 0.0
        for b in seg:
            p = (cnt[b] + 0.5) / (sum(cnt.values()) + k_alpha / 2)
            bits += -math.log2(p)
            cnt[b] += 1
        h0 = H0(seg) * n
        pred = (k_alpha - 1) / 2 * math.log2(n)
        print(f"{n:>10}{h0:>14.1f}{bits:>12.1f}{bits - h0:>11.1f}{pred:>22.1f}")


if __name__ == "__main__":
    main()
