"""
real_quant_curve.py — Qwen3-1.7B 的真实 PTQ 曲线（用原始 per-row scale 对齐）。

上一版错在哪：我用 step = s·σ_row 重建网格。但模型是 INT4 的，权重已经落在
`code × scale_row` 上；σ_row 和 scale_row 不是同一个数，所以重新量化等于把权重
挪到**另一个网格**上，误差顶到半个步长 —— ppl 直接炸到 1e5，测的不是压缩，是错位。

正确做法：直接用 IR 里的 `/scale`（每行一个），
    step_row = κ · scale_row
κ=1 时重构是**恒等**（可当 sanity check），κ>1 才是真正的"降有效电平数"。

K=8（码字 -8..7，16 电平），与部署格式一致。
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_qwen import DEV, IRFile, Tok, load, ppl  # noqa: E402

K = 8
MOD = dict(q="self_attn.q_proj", k="self_attn.k_proj", v="self_attn.v_proj",
           o="self_attn.o_proj", gate="mlp.gate_proj", up="mlp.up_proj", down="mlp.down_proj")


@torch.no_grad()
def q_kappa(w, scale_row, kappa, dz=1.0):
    """用原始 scale 对齐的网格做量化。返回 (重构, 码流熵, 有效电平数)。"""
    out = w.shape[0]
    r = w.reshape(out, -1).float()
    step = (kappa * scale_row.abs()).reshape(-1, 1)   # scale 可能为负（符号已烘进权重）
    t = r.abs() / step
    k = torch.clamp(torch.floor(t - dz / 2.0) + 1.0, 0.0, float(K))
    mag = torch.where(k > 0, (dz / 2.0 + k - 0.5) * step, torch.zeros_like(t))
    rec = torch.sign(r) * mag
    kk = (torch.where(r < 0, -k, k) + K).long().reshape(-1)
    cnt = torch.bincount(kk, minlength=2 * K + 1).float()
    pr = (cnt / cnt.sum())
    pr = pr[pr > 0].cpu().numpy()
    H = float(-(pr * np.log2(pr)).sum())
    return rec.reshape(w.shape), H, int((cnt > 0).sum())


def main():
    ir = IRFile()
    tok = Tok()
    m, _, _ = load()
    m = m.to(DEV).to(torch.bfloat16)
    p0, _ = ppl(m, tok)
    print(f"基线 ppl = {p0:.3f}\n")
    del m
    torch.cuda.empty_cache()

    # 预算每个张量的行 scale
    def scales(mdl):
        d = {}
        for i, l in enumerate(mdl.layers):
            for nm, mod in MOD.items():
                cn = f"self.model.layers.{i}.{mod}.weight/scale"
                d[f"layers.{i}.{nm}.weight"] = torch.from_numpy(
                    ir.raw_const(cn).astype(np.float32).reshape(-1, 1)).to(DEV).to(torch.bfloat16)
        return d

    print(f"{'kappa':>6}{'有效电平':>9}{'码流熵':>8}{'体积GB':>8}{'ppl':>10}{'Δ':>8}")
    out = []
    for kappa in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0):
        m2, _, _ = load()
        m2 = m2.to(DEV).to(torch.bfloat16)
        sc = scales(m2)
        tot = npar = 0
        lev = 0
        for n, p in m2.named_parameters():
            if n not in sc:
                continue
            rec, H, nl = q_kappa(p.data, sc[n], kappa)
            tot += H * p.numel()
            npar += p.numel()
            lev = max(lev, nl)
            p.data.copy_(rec.to(torch.bfloat16))
        bpw = tot / npar
        p, _ = ppl(m2, tok)
        gb = npar * bpw / 8 / 1e9
        out.append(dict(kappa=kappa, bpw=bpw, levels=lev, GB=gb, ppl=p))
        print(f"{kappa:>6.1f}{lev:>9d}{bpw:>8.3f}{gb:>8.3f}{p:>10.2f}{(p/p0-1)*100:>+7.1f}%",
              flush=True)
        del m2, sc
        torch.cuda.empty_cache()

    json.dump(dict(baseline=p0, points=out),
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dz",
                                "real_ptq_curve.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
