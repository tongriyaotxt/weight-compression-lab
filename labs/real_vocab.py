"""
real_vocab.py — 词表侧（embed + lm_head）能不能靠"降精度"而不是"熵编码"来压。

上一轮在 Qwen3-8B 上已经证明：u8 的 embed/lm_head 用无损手段最多省 1.3%
（零点、行内差分、每行一张表、低秩主子空间全部无效，H=7.19 已接近满熵）。

那剩下唯一的路就是**有损**：把 8 bit 降到 4/3/2 bit，看困惑度掉多少。
embedding 是查表，理论上对精度不敏感（每个 token 的方向只要大致对）。

本脚本在真模型（Qwen3-1.7B，ppl=18.47）上测：
    bits ∈ {8, 6, 5, 4, 3, 2} × 只量化 embedding（head 同步）
输出 (bits, 体积, 码流熵, ppl)。
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_qwen import DEV, Tok, load, ppl  # noqa: E402


@torch.no_grad()
def q_row_levels(w, bits):
    """按行、按 2^bits 个电平做对称量化（mid-tread）。"""
    levels = (1 << bits) - 1
    half = (levels - 1) / 2.0
    r = w.reshape(w.shape[0], -1).float()
    sig = r.std(dim=1, keepdim=True).clamp(min=1e-8)
    amax = r.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    # 15 电平用 max 裁剪时 step = amax/half
    step = amax / half
    q = torch.clamp(torch.round(r / step), -half, half)
    rec = q * step
    idx = (q + half).long().reshape(-1)
    cnt = torch.bincount(idx, minlength=int(levels)).float()
    p = (cnt / cnt.sum())[cnt > 0]
    H = float(-(p * torch.log2(p)).sum())
    return rec.reshape(w.shape), H, step * half


def main():
    tok = Tok()
    m, _, _ = load()
    m = m.to(DEV).to(torch.bfloat16)
    p0, _ = ppl(m, tok)
    n = m.emb.weight.numel()
    print(f"基线 ppl = {p0:.3f}   embedding {n/1e6:.0f}M 参数"
          f"（8 bit 名义 {n/1e9:.3f} GB，含 scale/zero_point）\n")
    print(f"{'bits':>5}{'码流熵':>9}{'体积MB':>9}{'ppl':>9}{'Δ':>9}")
    out = []
    for b in (8, 6, 5, 4, 3, 2):
        m2, _, _ = load()
        m2 = m2.to(DEV).to(torch.bfloat16)
        rec, H, amax = q_row_levels(m2.emb.weight.data, b)
        m2.emb.weight.data.copy_(rec.to(torch.bfloat16))
        m2.head.weight.data.copy_(rec.to(torch.bfloat16))   # tied
        p, _ = ppl(m2, tok)
        mb = n * H / 8 / 1e6
        out.append(dict(bits=b, H=H, MB=mb, ppl=p))
        print(f"{b:>5}{H:>9.3f}{mb:>9.1f}{p:>9.2f}{(p/p0-1)*100:>+8.1f}%", flush=True)
        del m2
        torch.cuda.empty_cache()
    json.dump(dict(baseline=p0, points=out),
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dz",
                                "real_vocab.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
