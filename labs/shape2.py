"""
shape2.py — 修正版：不是"非高斯"，是"稀疏"。

上一版的教训：在相同**方差**下比较微分熵是误导的，必须在相同**相对失真**下比较。
拉普拉斯在 D/σ²=0.02 时只省 0.06 bit，而不是微分熵暗示的 0.83 bit。
真正有杠杆的是 spike（精确零），因为它同时降低熵和方差。

扫一遍稀疏度 → 给出一条"稀疏度换比特"的曲线。
"""

import numpy as np

TARGET_D = 0.02029
GRID = 600_001
X = np.linspace(-8, 8, GRID)
DX = X[1] - X[0]


def spike_pdf(pi, sigma_tail=None):
    s = sigma_tail if sigma_tail else np.sqrt(1 / (1 - pi))
    p = (1 - pi) * np.exp(-X ** 2 / (2 * s * s)) / (np.sqrt(2 * np.pi) * s)
    p = p * DX
    p[GRID // 2] += pi
    p /= p.sum()
    m = (p * X).sum()
    v = (p * (X - m) ** 2).sum()
    p2 = np.interp(X * np.sqrt(v), X, p)
    p2 = np.maximum(p2, 0)
    return p2 / p2.sum()


def ternary_pdf(pi0=0.5):
    """三值: ±a 各 (1-pi0)/2，0 处 pi0"""
    a = np.sqrt(1 / (1 - pi0))
    p = np.zeros_like(X)
    p[GRID // 2] = pi0
    for sgn in (-1, 1):
        j = np.searchsorted(X, sgn * a)
        p[min(max(j, 0), GRID - 1)] += (1 - pi0) / 2
    p /= p.sum()
    v = (p * X ** 2).sum()
    p2 = np.interp(X * np.sqrt(v), X, p)
    p2 = np.maximum(p2, 0)
    return p2 / p2.sum()


def lloyd(pdf, L, iters=40):
    c = np.linspace(-3, 3, L)
    for _ in range(iters):
        b = np.concatenate([[-np.inf], (c[1:] + c[:-1]) / 2, [np.inf]])
        idx = np.clip(np.searchsorted(b, X) - 1, 0, L - 1)
        w = np.bincount(idx, weights=pdf, minlength=L) + 1e-300
        m = np.bincount(idx, weights=pdf * X, minlength=L)
        c = np.where(w > 1e-12, m / w, c)
    b = np.concatenate([[-np.inf], (c[1:] + c[:-1]) / 2, [np.inf]])
    idx = np.clip(np.searchsorted(b, X) - 1, 0, L - 1)
    D = float((pdf * (X - c[idx]) ** 2).sum())
    w = np.bincount(idx, weights=pdf, minlength=L)
    pr = w / w.sum()
    pr = pr[pr > 0]
    return D, float(-(pr * np.log2(pr)).sum())


def h_at(pdf, target=TARGET_D):
    prev = None
    for L in range(2, 200):
        D, H = lloyd(pdf, L)
        if D <= target:
            if prev is None:
                return H
            D0, H0 = prev
            t = (D0 - target) / (D0 - D)
            return H0 + t * (H - H0)
        prev = (D, H)
    return None


def main():
    print("=" * 72)
    print(f"相同相对失真 D/σ²={TARGET_D} 下，spike 比例 → 所需熵")
    print("=" * 72)
    print(f"  {'分布':<26}{'熵(bit)':>10}{'vs 纯高斯':>12}{'体积':>10}")
    print("  " + "-" * 58)
    base = None
    for pi in (0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9):
        H = h_at(spike_pdf(pi) if pi > 0 else spike_pdf(1e-9))
        if base is None:
            base = H
        lab = "高斯（现状）" if pi == 0 else f"spike {int(pi*100)}% 精确零"
        print(f"  {lab:<26}{H:>10.4f}{H-base:>+12.4f}{8.2*H/4:>9.2f}GB")

    Ht = h_at(ternary_pdf(0.5))
    print(f"  {'三值 (50% 零, ±a)':<26}{Ht:>10.4f}{Ht-base:>+12.4f}{8.2*Ht/4:>9.2f}GB")

    print()
    print("=" * 72)
    print("换算：Qwen3-8B（8.2B 参数）")
    print("=" * 72)
    print(f"  {'方案':<30}{'bpw':>8}{'体积':>10}")
    print("  " + "-" * 48)
    rows = [("现状 INT4 均匀 + max 裁剪", 4.00),
            ("+ 熵编码（实测）", 2.96),
            ("+ 最优标量量化 ECSQ", base),
            ("+ spike 50% 精确零", h_at(spike_pdf(0.5))),
            ("+ spike 70% 精确零", h_at(spike_pdf(0.7))),
            ("+ 三值 50% 零", Ht)]
    for name, b in rows:
        print(f"  {name:<30}{b:>8.3f}{8.2*b/4:>9.2f}GB")

    print()
    print("=" * 72)
    print("关键判断")
    print("=" * 72)
    g = h_at(spike_pdf(0.7))
    print(f"  从纯高斯到 70% 零：{g-base:+.3f} bit  （{100*(base-g)/base:.1f}%）")
    print(f"  注意：这是**相同失真**下的比较，稀疏度是真的在换比特。")
    print(f"  但它必须由训练产生 —— 事后剪枝做不到（剪掉的就是信息）。")


if __name__ == "__main__":
    main()
