"""
dz_calib.py — 事后标定：给定一个已训练好的模型，扫描 (dz, step)，
画出真实的 (码流熵, 部署后 val bpc) 前沿。

这是与训练无关的那一半问题：
  ECSQ 理论（Gish–Pierce 1979 / Farvardin–Modestino 1985）说，
  对尖峰源，最优标量量化器的中心 bin 必须比外侧宽（dz > 1）。
  但那个结论是在**连续分布假设**下、以解析 R(D) 得到的。
  这里是真实的训练权重 + 真实的任务损失，直接测。

读法：
  对每个 dz，扫 step 得到一条 (bits, bpc) 曲线；
  在**同一 bpc** 处比较不同 dz 的 bits —— 差多少就是死区的净收益。
"""

import glob
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dz_train import DEV, K, GPT, FlatQ, batches, catw, corpus, huff_avg  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dz")


@torch.no_grad()
def set_q(qd, dz_val, kappa, base_step):
    """固定 dz 与 step=kappa*base_step。"""
    qd.log_step.copy_(torch.log(base_step * kappa))
    # rho 使 1+softplus(rho)=dz
    rho = torch.log(torch.expm1(torch.clamp(torch.tensor(dz_val - 1.0), min=1e-6)))
    qd.rho.fill_(float(rho))


@torch.no_grad()
def bits_of(qd, base_step):
    flat = catw(qd)
    step = qd.step()[qd.el]
    dz = qd.dz()[qd.el]
    t = flat.abs() / step
    k = torch.clamp(torch.floor(t - dz / 2.0) + 1.0, 0.0, float(K))
    k = (torch.where(flat < 0, -k, k) + K).long()
    tot = 0.0
    for i in range(len(qd.mods)):
        s, n = qd.starts[i].item(), qd.sizes[i]
        cnt = torch.bincount(k[s:s + n], minlength=2 * K + 1).float()
        p = (cnt / cnt.sum()).cpu().numpy()
        nz = p[p > 0]
        tot += float(-(nz * np.log2(nz)).sum()) * n
    return tot / qd.n


def load(path, ctx=128, d=160, L=4):
    ck = torch.load(path, map_location=DEV, weights_only=False)
    model = GPT(d=d, L=L, ctx=ctx).to(DEV)
    model.load_state_dict(ck["model"])
    model.eval()
    qd = FlatQ([m for m in model.modules() if isinstance(m, torch.nn.Linear)]).to(DEV)
    with torch.no_grad():
        qd.log_step.copy_(ck["log_step"].to(DEV))
        qd.rho.copy_(ck["rho"].to(DEV))
    return model, qd, ck["args"]


def main():
    GC = "--gc" in sys.argv
    ap_paths = sorted(glob.glob(os.path.join(OUT, "models", "*.pt")))
    sel = [a for a in sys.argv[1:] if not a.startswith("--")]
    if sel:
        ap_paths = [p for p in ap_paths if any(a in os.path.basename(p) for a in sel)]
    raw = np.frombuffer(corpus(), dtype=np.uint8)
    split = int(len(raw) * 0.95)
    va = raw[split:]
    vs = list(batches(va, 128, 24, 24, seed=99))

    DZS = [1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    KAPS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 1.7]

    for p in ap_paths:
        model, qd, args = load(p)
        base = qd.step().detach().clone()
        print(f"\n=== {os.path.basename(p)}  (mode={args.get('mode')} lam={args.get('lam')} dzlearn={args.get('dzlearn')}) ===")
        rows = []
        for dz in DZS:
            pts = []
            for kap in KAPS:
                set_q(qd, dz, kap, base)
                b = bits_of(qd, base)
                qd_qt = {m: None for m in qd.mods}
                flat = catw(qd)
                step = qd.step()[qd.el]
                dzv = qd.dz()[qd.el]
                t = flat.abs() / step
                k = torch.clamp(torch.floor(t - dzv / 2.0) + 1.0, 0.0, float(K))
                mag = torch.where(k > 0, (dzv / 2.0 + k - 0.5) * step, torch.zeros_like(t))
                qflat = torch.sign(flat) * mag
                if GC and kap > 0:
                    # 尺度补偿：recon 退回模型原生 step 的尺度，只让"分辨率"变化
                    qflat = qflat * (torch.ones_like(step) / kap)
                W = {}
                for i, m in enumerate(qd.mods):
                    s, n = qd.starts[i].item(), qd.sizes[i]
                    W[m] = qflat[s:s + n].view(qd.shapes[i])
                tot = 0.0
                for x, y in vs:
                    tot += F.cross_entropy(model(x, W).reshape(-1, 256), y.reshape(-1)).item()
                pts.append((b, tot / len(vs) / math.log(2), kap))
            rows.append((dz, pts))
            best = min(pts, key=lambda z: z[1] + 0.0)
            print(f"  dz={dz:<5} " + "  ".join(f"{b:5.3f}b/{q:5.3f}" for b, q, _ in pts))

        # 等质量点：以 dz=1 的最低 bpc 为参考，找每档 dz 在 bpc<=ref+0.02 下的最小 bits
        ref_q = min(q for _, pts in rows[:1] for _, q, _ in pts)
        print(f"  -- 等质量前沿（bpc <= {ref_q:.4f}+0.02）--")
        for dz, pts in rows:
            ok = [(b, q, k) for b, q, k in pts if q <= ref_q + 0.02]
            if ok:
                b = min(ok)[0]
                print(f"     dz={dz:<5} min bits = {b:.4f}")


if __name__ == "__main__":
    main()
