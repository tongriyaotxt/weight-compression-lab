"""
middle_out_probe.py — "middle-out" 到底行不行？用可算的代价函数量一下。

四个编码方案，同一个语料，同一个自适应估计器（KT + 插值回退）：

  A  L->R      顺序左到右，1 阶上下文
  B  R->L      顺序右到左，1 阶上下文
  C  MID->OUT  从正中间开始向两侧同时推进，共享一个 1 阶自适应模型
  D  2-PASS    真正的“双边”：先传偶数位（1 阶），再用 [左,右] 双邻居条件补奇数位

A/B/C 的差别只是遍历顺序；D 才是真正多拿了信息的那一个。
"""

import glob
import math
import os
import sys
import sysconfig
from collections import Counter, defaultdict

K = 256
ALPHA = 1.0  # 插值回退强度


# ---------------------------------------------------------------- estimator
class Interp:
    """插值回退自适应估计器：从短上下文往长上下文，逐层插值。"""

    def __init__(self):
        self.tab = defaultdict(Counter)
        self.base = Counter()
        self.n = 0

    def prob(self, ctxs, x):
        p = (self.base[x] + 0.5) / (self.n + K / 2)
        for ctx in ctxs:  # 由短到长
            c = self.tab[ctx]
            tot = sum(c.values())
            if tot == 0:
                continue
            lam = tot / (tot + ALPHA)
            p = lam * (c[x] / tot) + (1 - lam) * p
        return p

    def update(self, ctxs, x):
        for ctx in ctxs:
            self.tab[ctx][x] += 1
        self.base[x] += 1
        self.n += 1


def bits_for(order_of, data):
    m = Interp()
    bits = 0.0
    for i, (ctxs, x) in enumerate(order_of):
        p = m.prob(ctxs, x)
        bits += -math.log2(max(p, 1e-12))
        m.update(ctxs, x)
    return bits


# ---------------------------------------------------------------- schemes
def scheme_lr(data):
    return [((data[i - 1],) if i else (), x) for i, x in enumerate(data)]


def scheme_rl(data):
    rev = data[::-1]
    return [((rev[i - 1],) if i else (), x) for i, x in enumerate(rev)]


def scheme_midout(data):
    """从中心向两侧交替推进；共享同一个自适应模型。"""
    n = len(data)
    seen = bytearray(n)          # 1 = 已编码（邻居已知）
    order = []
    lo, hi = (n - 1) // 2, (n - 1) // 2 + 1
    seen[lo] = 1
    order.append(lo)
    left, right = lo - 1, hi
    turn = 0
    while left >= 0 or right < n:
        if turn % 2 == 0 and left >= 0:
            order.append(left)
            seen[left] = 1
            left -= 1
        elif right < n:
            order.append(right)
            seen[right] = 1
            right += 1
        turn += 1

    # 遍历顺序定好后，按“已见邻居”给上下文
    out = []
    seen = bytearray(n)
    for idx in order:
        ctx = None
        if idx - 1 >= 0 and seen[idx - 1]:
            ctx = (data[idx - 1],)          # 优先左邻居
        elif idx + 1 < n and seen[idx + 1]:
            ctx = (data[idx + 1],)
        out.append(((ctx,) if ctx else (), data[idx]))
        seen[idx] = 1
    return out


def scheme_lr2(data):
    """对照组：普通 2 阶，看 D 的收益是不是只来自“上下文更多”。"""
    return [((((data[i - 1],), (data[i - 2], data[i - 1])) if i >= 2
              else ((data[i - 1],) if i else ())), x)
            for i, x in enumerate(data)]


def scheme_twopass(data):
    """D：偶数位 → 奇数位（用左右两个已知邻居）。"""
    n = len(data)
    evens = data[0::2]
    odds = data[1::2]
    out = []
    prev = None
    for x in evens:                       # pass 1: 偶数位，1 阶
        out.append(((prev,) if prev is not None else (), x))
        prev = x
    for j, x in enumerate(odds):          # pass 2: 奇数位，双邻居
        i = 2 * j + 1
        l = data[i - 1]
        r = data[i + 1] if i + 1 < n else 0
        out.append((((l,), (l, r)), x))   # 由短到长：单邻居 -> 双邻居
    return out


# ---------------------------------------------------------------- corpus
def corpus(target=100_000):
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


def main():
    data = corpus()
    n = len(data)
    print(f"corpus: {n} bytes of Python stdlib  ({8*n/8/1024:.0f} KiB)\n")

    rows = []
    for name, fn in [
        ("A  L->R sequential (order-1)", scheme_lr),
        ("B  R->L sequential (order-1)", scheme_rl),
        ("C  MID->OUT shared model", scheme_midout),
        ("D  2-PASS even/odd (bi-dir ctx)", scheme_twopass),
        ("E  L->R order-2 (control)", scheme_lr2),
    ]:
        order = fn(data)
        b = bits_for(order, data)
        rows.append((name, b, b / n))

    base = rows[0][1]
    print(f"{'scheme':<34}{'total bits':>14}{'bpc':>9}{'vs A':>10}")
    print("-" * 67)
    for name, b, bpc in rows:
        print(f"{name:<34}{b:>14.0f}{bpc:>9.4f}{100*(b-base)/base:>9.2f}%")

    print()
    print(f"{'gap':<34}{'delta bpc':>12}{'%':>9}")
    print("-" * 56)
    print(f"{'B - A  (direction only)':<34}{rows[1][2]-rows[0][2]:>12.4f}"
          f"{100*(rows[1][1]-base)/base:>8.2f}%")
    print(f"{'C - A  (traversal only)':<34}{rows[2][2]-rows[0][2]:>12.4f}"
          f"{100*(rows[2][1]-base)/base:>8.2f}%")
    print(f"{'D - A  (extra neighbours)':<34}{rows[3][2]-rows[0][2]:>12.4f}"
          f"{100*(rows[3][1]-base)/base:>8.2f}%")


if __name__ == "__main__":
    main()
