"""
embed_link.py — 上一轮只做了"整块哈希比对"（结论：不是同一份）。
但哈希只排除"完全相同"。真正的问题是：**embed 和 lm_head 作为两张 151936x4096 的
token<->方向 映射表，在功能上是否冗余？**

若 lm_head ≈ M @ embed，那么 622 MB 的 lm_head 可以换成 4096x4096 的 M
（4 bit 下 ~8 MB）+ 残差。这是文件里最大的一块单一可攻目标（1187 MiB = 26.4%）。

三组测量，逐级放宽：
  A  逐行余弦（按 token 对齐）           —— 最严格：同一个 token 的方向是否一致
  B  线性映射 R²（ridge，4096x4096）     —— 是否能用一个线性变换互相表达
  C  映射后残差的熵（接上 QEC 账本）      —— 即使 R² 不满，残差还剩多少 bit

全部只读，mmap 访问。
"""

import mmap
import os
import sys
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
                                off=int(d.get("offset", 0)), size=int(d.get("size")))
fh = open(os.path.join(IR, "openvino_model.bin"), "rb")
mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)

E_NAME = "self.model.embed_tokens.weight"
H_NAME = "self.lm_head.weight"


def deq(name, rows):
    """按行反量化：W[i,:] = (q - zp[i]) * scale[i]"""
    c = C[name]
    R, D = c["shape"]
    sp = C[name + "/scale"]
    zp = C[name + "/zero_point"]
    s = np.frombuffer(mm, np.float16, int(np.prod(sp["shape"])), sp["off"]).astype(np.float32)
    z = np.frombuffer(mm, np.uint8, int(np.prod(zp["shape"])), zp["off"]).astype(np.float32)
    q = np.frombuffer(mm, np.uint8, R * D, c["off"]).reshape(R, D)
    out = np.empty((len(rows), D), np.float32)
    for j, r in enumerate(rows):
        out[j] = (q[r].astype(np.float32) - z[r]) * s[r]
    return out, (s, z)


def main():
    R, D = C[E_NAME]["shape"]
    print("=" * 76)
    print(f"embed {C[E_NAME]['shape']}   lm_head {C[H_NAME]['shape']}   hidden={D}")
    print("=" * 76)

    rng = np.random.default_rng(0)
    n = 16384                      # 一半训练 / 一半测试，否则 4096 维拟合 4096 样本必然 R²=1
    rows = np.sort(rng.choice(R, n, replace=False))

    X, (sx, zx) = deq(E_NAME, rows)
    Y, (sy, zy) = deq(H_NAME, rows)

    print(f"\nscale 分布: embed median={np.median(sx):.6f}  lm_head median={np.median(sy):.6f}")
    print(f"行范数   : embed median={np.median(np.linalg.norm(X,axis=1)):.3f}  "
          f"lm_head median={np.median(np.linalg.norm(Y,axis=1)):.3f}")

    # ---------------- A 逐行余弦 ----------------
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
    cos = (Xn * Yn).sum(1)
    print("\n" + "=" * 76)
    print("A  逐行余弦（同一 token 的 embed 方向 vs lm_head 方向）")
    print("=" * 76)
    print(f"  均值 {cos.mean():+.4f}   中位 {np.median(cos):+.4f}   "
          f"std {cos.std():.4f}   |cos|>0.5 占比 {100*(np.abs(cos)>0.5).mean():.2f}%")
    print(f"  解析零基线 1/sqrt({D}) = {1/np.sqrt(D):.4f}")
    # 独立随机行的余弦作对照
    perm = rng.permutation(n)
    cos0 = (Xn * Yn[perm]).sum(1)
    print(f"  随机配对基线: 均值 {cos0.mean():+.4f}  std {cos0.std():.4f}")

    # ---------------- B 线性映射 R²（分半，样本外） ----------------
    print("\n" + "=" * 76)
    print("B  lm_head ≈ M @ embed ?  （ridge，4096x4096，训练/测试分半）")
    print("=" * 76)
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        half = n // 2
        tx = torch.from_numpy(X[:half]).to(dev)
        ty = torch.from_numpy(Y[:half]).to(dev)
        vx = torch.from_numpy(X[half:]).to(dev)
        vy = torch.from_numpy(Y[half:]).to(dev)
        mx, my = tx.mean(0, keepdim=True), ty.mean(0, keepdim=True)
        tx, ty = tx - mx, ty - my
        vx, vy = vx - mx, vy - my
        for lam_mult in (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0):
            lam = lam_mult * float((tx * tx).mean()) + 1e-12
            G = tx.T @ tx + lam * torch.eye(D, device=dev)
            M = torch.linalg.solve(G, tx.T @ ty)
            pred = vx @ M
            r2 = 1 - float(((vy - pred) ** 2).sum()) / float((vy ** 2).sum())
            predi = tx @ M
            r2i = 1 - float(((ty - predi) ** 2).sum()) / float((ty ** 2).sum())
            print(f"  λ={lam_mult:<7g}  train R²={r2i:.4f}   **test R²={r2:.4f}**")
        # 取最保守的一档做后续分析
        lam = 1e-1 * float((tx * tx).mean()) + 1e-12
        G = tx.T @ tx + lam * torch.eye(D, device=dev)
        M = torch.linalg.solve(G, tx.T @ ty)
        pred = vx @ M
        r2 = 1 - float(((vy - pred) ** 2).sum()) / float((vy ** 2).sum())
        resid = (vy - pred).cpu().numpy()
        rs = float(resid.std())
        ys = float(vy.std())
        print(f"\n  选定 λ={lam:.3e}  test R²={r2:.4f}")
        print(f"  残差 σ = {rs:.6f}   原 σ = {ys:.6f}   比 = {rs/ys:.4f}")
        # 位深等价：残差再用同样位深量化，需要多少 bit 才能达到同等失真
        print(f"  → 残差所需 bit = {np.log2(ys/rs):.3f} bit 少于原张量（按 R(D) 等比例）")
        sv = torch.linalg.svdvals(M.float()).cpu().numpy()
        e = np.cumsum(sv ** 2) / np.sum(sv ** 2)
        for k in (16, 64, 128, 256, 512, 1024):
            print(f"    M 的 rank-{k:<4} 捕获 {100*e[k-1]:.2f}% 能量")
    except Exception as ex:
        print(f"  torch 路径失败: {ex}")

    # ---------------- B2 子空间重叠（低维，不会过拟合） ----------------
    print("\n" + "=" * 76)
    print("B2 embed 与 lm_head 是否张成同一个子空间？（留出评估）")
    print("=" * 76)
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        half = n // 2
        tx = torch.from_numpy(X[:half]).to(dev)
        vx = torch.from_numpy(X[half:]).to(dev)
        vy = torch.from_numpy(Y[half:]).to(dev)
        tx = tx - tx.mean(0, keepdim=True)
        vx = vx - vx.mean(0, keepdim=True)
        vy = vy - vy.mean(0, keepdim=True)
        # embed 的 PCA 基（只用训练半）
        U, S, Vh = torch.linalg.svd(tx, full_matrices=False)
        for k in (32, 64, 128, 256, 512):
            B = Vh[:k].T                      # (4096, k) —— 特征空间的主方向
            e_y = float(((vy @ B) ** 2).sum() / (vy ** 2).sum())
            e_x = float(((vx @ B) ** 2).sum() / (vx ** 2).sum())
            print(f"  k={k:<4} lm_head 能量落在 embed 前 k 个主方向: {100*e_y:6.2f}%"
                  f"   （embed 自身 {100*e_x:5.2f}%，随机基线 {100*k/D:.2f}%）")
        print("  解读：若 lm_head 那列 ≈ 随机基线，则两表在几何上完全独立。")
    except Exception as ex:
        print(f"  子空间测试失败: {ex}")

    # ---------------- C 若存 M + 残差，账本怎么变 ----------------
    print("\n" + "=" * 76)
    print("C  账本")
    print("=" * 76)
    print(f"  lm_head 现状（u8 + 行 scale）: {C[H_NAME]['size']/2**20:.1f} MiB")
    print(f"  若 M(4096x4096)@4bit + 残差: M={4096*4096*0.5/2**20:.1f} MiB；残差按上面 σ 比定 bpw")


if __name__ == "__main__":
    main()
