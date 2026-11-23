"""
ecsq_theory.py — 熵约束标量量化（ECSQ）在标准正态源上的最优死区比例。

为什么需要这条曲线：
  上一轮我用 R(D) 界算出"换分布只值 0.06 bit"，但那是连续最优。
  真正部署的是"死区 + 有限电平 + 熵编码"的 ECSQ，它的最优 dz 随码率变化：
      中高码率：dz* ≈ 1（这里量出来确实如此）
      极低码率：dz* > 1（中心 bin 该吞掉更多概率）
  这条曲线是本轮 NN 训练的对照预言。

模型：X ~ N(0,1)，量化器 = 中心 bin 宽 dz·Δ，外侧步长 Δ，每侧 K 个电平。
扫描 Δ 得到 (H, D) 曲线，取每个 H 下的最小 D，再跨 dz 比较。
"""

import numpy as np

K = 7
GRID = 200_001
X = np.linspace(-9, 9, GRID)
DX = X[1] - X[0]
PDF = np.exp(-0.5 * X ** 2) / np.sqrt(2 * np.pi)
EX2 = float((X ** 2 * PDF).sum() * DX)


def stats(delta, dz):
    t = np.abs(X) / delta
    k = np.clip(np.floor(t - dz / 2.0) + 1.0, 0, K)
    mag = np.where(k > 0, (dz / 2.0 + k - 0.5) * delta, 0.0)
    rec = np.sign(X) * mag
    p = np.bincount(k.astype(np.int32), weights=PDF * DX, minlength=K + 1)
    p = p / p.sum()
    nz = p[p > 0]
    H = float(-(nz * np.log2(nz)).sum())
    D = float(((X - rec) ** 2 * PDF).sum() * DX) / EX2
    return H, D


DELTAS = np.exp(np.linspace(np.log(2e-3), np.log(2.0), 320))


def curve(dz):
    return [(stats(d, dz)[0], stats(d, dz)[1], d) for d in DELTAS]


def minD(pts, Ht, tol=0.04):
    c = [D for H, D, _ in pts if abs(H - Ht) <= tol]
    return min(c) if c else None


def H_at_D(pts, Dt, tol=2e-4):
    """在给定 dz 的曲线上找能达到 Dt 所需的最小 H。"""
    c = [H for H, D, _ in pts if abs(D - Dt) <= tol]
    return min(c) if c else None


def main():
    DZS = [1.0, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]
    C = {dz: curve(dz) for dz in DZS}
    print("标准正态源，15 电平死区量化器 + 熵编码")
    print(f"{'H':>5}{'dz*':>7}{'D*':>10}{'D(dz=1)':>11}{'失真增益dB':>12}{'等效省bit':>11}")
    for Ht in (3.2, 3.0, 2.5, 2.0, 1.6, 1.3, 1.0, 0.8, 0.6, 0.45, 0.3):
        best = None
        for dz in DZS:
            D = minD(C[dz], Ht)
            if D is not None and (best is None or D < best[1]):
                best = (dz, D)
        if best is None:
            continue
        dz, D = best
        D1 = minD(C[1.0], Ht)
        save = H_at_D(C[1.0], D)
        gdb = 10 * np.log10(D1 / D) if D1 and D else float("nan")
        sv = (save - Ht) if save is not None else float("nan")
        print(f"{Ht:>5.2f}{dz:>7.2f}{D:>10.5f}{D1:>11.5f}{gdb:>12.2f}{sv:>11.2f}")
    print("\n注：等效省 bit = 在 dz=1 曲线上达到同样失真所需的码率，减去当前码率。负值表示死区无益。")


if __name__ == "__main__":
    main()
