"""
shape.py — 核心问题：能不能通过改变权重的"分布形状"来省 bit？

逻辑链：
  1. 实测：权重是 i.i.d. 高斯，而高斯是给定方差下**熵最大**的分布
  2. 高斯的来源：weight decay + SGD 噪声 = Ornstein-Uhlenbeck 过程，
     其平稳分布恰好是高斯。所以"高斯"是动力学的必然，不是巧合
  3. 推论：任何非二次正则（L1/L0/elastic net）都会把平稳分布改成更低熵的形状
  4. 量化：在**相同相对失真 D/σ²** 下，比较不同分布所需的熵

如果拉普拉斯能省 0.8 bit，那"训练时就让它可压"就是一个真实的方向。
"""

import numpy as np

TARGET_D = 0.02029      # 实测：OpenVINO INT4 的量化失真（归一化到 σ²=1）
GRID = 400_001
X = np.linspace(-7, 7, GRID)


def make_pdf(kind):
    if kind == "gauss":
        p = np.exp(-X ** 2 / 2)
    elif kind == "laplace":
        b = 1 / np.sqrt(2)
        p = np.exp(-np.abs(X) / b)
    elif kind == "uniform":
        a = np.sqrt(3)
        p = (np.abs(X) <= a).astype(float)
    elif kind == "spike0.3":
        s = np.sqrt(1 / 0.7)
        p = 0.7 * np.exp(-X ** 2 / (2 * s * s))
        p[GRID // 2] += 0.3 / (X[1] - X[0]) * 1.0
    elif kind == "spike0.5":
        s = np.sqrt(1 / 0.5)
        p = 0.5 * np.exp(-X ** 2 / (2 * s * s))
        p[GRID // 2] += 0.5 / (X[1] - X[0]) * 1.0
    elif kind == "student3":
        p = (1 + X ** 2 / 3) ** (-2.0)
    p = p / p.sum()
    # 归一化到方差 1
    m = (p * X).sum()
    v = (p * (X - m) ** 2).sum()
    p = np.interp(X * np.sqrt(v) + m, X, p)
    p = np.maximum(p, 0)
    p /= p.sum()
    return p


def lloyd(pdf, L, iters=80):
    c = np.linspace(np.percentile(X, 2), np.percentile(X, 98), L)
    for _ in range(iters):
        b = np.concatenate([[-np.inf], (c[1:] + c[:-1]) / 2, [np.inf]])
        idx = np.clip(np.searchsorted(b, X) - 1, 0, L - 1)
        w = np.bincount(idx, weights=pdf, minlength=L) + 1e-300
        m = np.bincount(idx, weights=pdf * X, minlength=L)
        c = np.where(w > 1e-12, m / w, c)
    b = np.concatenate([[-np.inf], (c[1:] + c[:-1]) / 2, [np.inf]])
    idx = np.clip(np.searchsorted(b, X) - 1, 0, L - 1)
    q = c[idx]
    D = float((pdf * (X - q) ** 2).sum())
    w = np.bincount(idx, weights=pdf, minlength=L)
    pr = w / w.sum()
    pr = pr[pr > 0]
    H = float(-(pr * np.log2(pr)).sum())
    return D, H, c


def h_at_target(pdf):
    """找出达到 TARGET_D 所需的最小熵（扫描电平数并插值）。"""
    rows = []
    for L in range(3, 130):
        D, H, c = lloyd(pdf, L, iters=25)
        rows.append((D, H))
        if D < TARGET_D * 0.5:
            break
    rows = np.array(rows)
    # D 随 L 单调下降，插值到 TARGET_D
    if rows[-1, 0] > TARGET_D:
        return None
    for i in range(len(rows) - 1):
        if rows[i, 0] >= TARGET_D >= rows[i + 1, 0]:
            t = (rows[i, 0] - TARGET_D) / (rows[i, 0] - rows[i + 1, 0])
            return rows[i, 1] + t * (rows[i + 1, 1] - rows[i, 1])
    return rows[-1, 1]


def main():
    print("=" * 76)
    print(f"在相同相对失真 D/σ² = {TARGET_D} 下，不同分布形状所需的熵")
    print("=" * 76)
    print(f"  {'分布':<22}{'熵(bit)':>10}{'vs 高斯':>12}{'@4bit 省':>12}")
    print("  " + "-" * 56)
    base = None
    for kind, label in [
        ("gauss", "高斯（weight decay）"),
        ("uniform", "均匀"),
        ("laplace", "拉普拉斯（L1）"),
        ("student3", "Student-t (ν=3)"),
        ("spike0.3", "spike-slab π=0.3"),
        ("spike0.5", "spike-slab π=0.5"),
    ]:
        p = make_pdf(kind)
        H = h_at_target(p)
        if H is None:
            print(f"  {label:<22} 未能达到目标失真")
            continue
        if base is None:
            base = H
        print(f"  {label:<22}{H:>10.4f}{H-base:>+12.4f}{4*(H-base)/4:>11.1f}%")

    print()
    print("=" * 76)
    print("对照：当前模型实测")
    print("=" * 76)
    print("  OpenVINO INT4（均匀量化 + max 裁剪）实测 H0 = 2.962（均值）/ 3.036（L18 up_proj）")
    print("  → 上表中'高斯'一行对应的是 ECSQ 最优（Lloyd-Max）的水平，")
    print("    比实测的均匀量化好约 0.2 bit，这就是'最优标量量化'的空间。")
    print()
    print("=" * 76)
    print("换算到模型体积")
    print("=" * 76)
    p = make_pdf("laplace")
    Hl = h_at_target(p)
    pg = make_pdf("gauss")
    Hg = h_at_target(pg)
    n = 8.2e9
    print(f"  Qwen3-8B 若权重是高斯 : {n*Hg/8/1e9:.2f} GB")
    print(f"  Qwen3-8B 若权重是拉普拉斯: {n*Hl/8/1e9:.2f} GB")
    print(f"  → 同样质量、同样方差，白省 {n*(Hg-Hl)/8/1e9:.2f} GB"
          f"（{100*(Hg-Hl)/Hg:.1f}%）")


if __name__ == "__main__":
    main()
