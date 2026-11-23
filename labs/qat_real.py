"""
qat_real.py — 真 Qwen3-1.7B 上的量化感知微调（固定网格 + 分层 fp32 主权重）。

背景（本轮测出来的两件事）：
  · κ=1（用 IR 原始 per-row scale）重量化是**恒等**，ppl 18.47 —— 量化管线可信
  · κ=1.5 → 2.44 bpw，ppl 39.6；κ=2.0 直接死。所以部署网格是个悬崖

QAT 要回答的：把 κ=1.5 的损伤（+114% ppl）能拉回多少？

显存策略：只把最后 N 层转成 fp32 主权重（N×54M×4B），其余层留 bf16 冻结。
bf16 主权重收不到 1e-6 量级的更新，所以必须 fp32 —— 这是上一轮没做成的原因。

用法：
  python labs/qat_real.py --kappa 1.5 --train_layers 8 --steps 200
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_qwen import DEV, IRFile, Tok, load, ppl, rope  # noqa: E402

K = 8
MOD = dict(q="self_attn.q_proj", k="self_attn.k_proj", v="self_attn.v_proj",
           o="self_attn.o_proj", gate="mlp.gate_proj", up="mlp.up_proj", down="mlp.down_proj")


def dzq(w, step, dz=1.0, clip_ste=True):
    t = w.abs() / step
    k = torch.clamp(torch.floor(t - dz / 2.0) + 1.0, 0.0, float(K))
    mag = torch.where(k > 0, (dz / 2.0 + k - 0.5) * step, torch.zeros_like(t))
    q = torch.sign(w) * mag
    out = w + (q - w).detach()
    if clip_ste:
        out = torch.where((t > float(K)).detach(), q.detach(), out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=float, default=1.5)
    ap.add_argument("--train_layers", type=int, default=8)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--dz", type=float, default=1.0)
    ap.add_argument("--tag", default="qat")
    args = ap.parse_args()

    ir = IRFile()
    tok = Tok()
    m, _, _ = load()
    m = m.to(DEV).to(torch.bfloat16)
    p_fp, _ = ppl(m, tok)

    # 每个 Linear 的 per-row step
    steps = {}
    for i, l in enumerate(m.layers):
        for nm, mod in MOD.items():
            s = ir.raw_const(f"self.model.layers.{i}.{mod}.weight/scale")
            steps[f"layers.{i}.{nm}.weight"] = torch.from_numpy(
                args.kappa * np.abs(s.astype(np.float32))).to(DEV).to(torch.bfloat16).clamp(min=1e-8)

    # 分层：最后 N 层转 fp32 主权重
    nl = len(m.layers)
    train_names = set()
    for i in range(nl - args.train_layers, nl):
        for nm in MOD:
            train_names.add(f"layers.{i}.{nm}.weight")
    params = []
    for n, p in list(m.named_parameters()):
        if n in train_names:
            parts = n.split(".")              # ["layers","27","q","weight"]
            mod = m
            for q in parts[:-1]:              # 走到 Linear 模块
                mod = mod[int(q)] if q.isdigit() else getattr(mod, q)
            newp = torch.nn.Parameter(p.data.float())
            setattr(mod, "weight", newp)      # 换掉 weight 本身
            params.append(newp)
    m = m.to(DEV)
    q_params = [p for p in params]
    print(f"基线 ppl(fp16前向) {p_fp:.3f}   kappa={args.kappa}   "
          f"训练层 {args.train_layers}/{nl}  可训参数 {sum(p.numel() for p in q_params)/1e6:.0f}M",
          flush=True)

    frozen = {}

    @torch.no_grad()
    def freeze_all():
        """冻结层直接**替换**成量化后的权重（丢掉原件，省一半显存）"""
        import gc
        for i, l in enumerate(m.layers):
            for nm in MOD:
                key = f"layers.{i}.{nm}.weight"
                if key in train_names:
                    continue
                mod = getattr(l, nm)
                w = mod.weight
                q = dzq(w.data, steps[key].to(w.dtype), args.dz).detach()
                mod.weight = torch.nn.Parameter(q, requires_grad=False)
                frozen[(i, nm)] = mod.weight.data
                del w, q
        gc.collect()
        torch.cuda.empty_cache()

    def fwd(ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        x = m.emb(ids)
        for i, l in enumerate(m.layers):
            dt = l.q.weight.dtype
            h = l.ln1(x).to(dt)
            def W(nm):
                key = f"layers.{i}.{nm}.weight"
                if (i, nm) in frozen:
                    return frozen[(i, nm)]
                return dzq(getattr(l, nm).weight, steps[key].to(dt), args.dz)
            wq, wk, wv, wo = W("q"), W("k"), W("v"), W("o")
            q = F.linear(h, wq).view(B, T, l.nh, l.hd).transpose(1, 2)
            k = F.linear(h, wk).view(B, T, l.nkv, l.hd).transpose(1, 2)
            v = F.linear(h, wv).view(B, T, l.nkv, l.hd).transpose(1, 2)
            q, k = l.qn(q), l.kn(k)
            q, k = rope(q, pos, dim=l.hd), rope(k, pos, dim=l.hd)
            k = k.repeat_interleave(l.nh // l.nkv, dim=1)
            v = v.repeat_interleave(l.nh // l.nkv, dim=1)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            a = a.transpose(1, 2).reshape(B, T, -1)
            x = x + F.linear(a, wo)
            h2 = l.ln2(x).to(dt)
            gu = F.silu(F.linear(h2, W("gate"))) * F.linear(h2, W("up"))
            x = x + F.linear(gu, W("down"))
        return F.linear(m.norm(x).to(m.head.weight.dtype), m.head.weight)

    freeze_all()
    # 先量一次"只量化不训练"的 ppl（用同一前向）
    ids_all = tok.encode(__import__("real_qwen").SAMPLE)
    x0 = torch.tensor([ids_all * 3], device=DEV)
    with torch.no_grad():
        tot, n = 0.0, 0
        for i in range(0, x0.shape[1] - 1, args.ctx):
            seg = x0[:, i:i + args.ctx + 1]
            if seg.shape[1] < 3:
                break
            lo = fwd(seg[:, :-1]).float()
            tot += F.cross_entropy(lo.reshape(-1, lo.shape[-1]), seg[:, 1:].reshape(-1),
                                   reduction="sum").item()
            n += seg.shape[1] - 1
    p_q = float(np.exp(tot / n))
    print(f"只量化不训练 ppl = {p_q:.2f}  ({(p_q/p_fp-1)*100:+.1f}%)", flush=True)

    data = torch.tensor([tok.encode(open(__file__, encoding="utf-8").read()[:6000]) * 4],
                        device=DEV)[0]
    opt = torch.optim.SGD(q_params, lr=args.lr)
    g = torch.Generator().manual_seed(0)
    t0 = time.time()
    for step in range(args.steps):
        i = int(torch.randint(0, max(1, len(data) - args.ctx - 1), (1,), generator=g))
        seg = data[i:i + args.ctx + 1][None]
        lo = fwd(seg[:, :-1]).float()
        loss = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), seg[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_params, 1.0)
        opt.step()
        if (step + 1) % 50 == 0:
            print(f"  step {step+1:4d} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)

    with torch.no_grad():
        tot, n = 0.0, 0
        for i in range(0, x0.shape[1] - 1, args.ctx):
            seg = x0[:, i:i + args.ctx + 1]
            if seg.shape[1] < 3:
                break
            lo = fwd(seg[:, :-1]).float()
            tot += F.cross_entropy(lo.reshape(-1, lo.shape[-1]), seg[:, 1:].reshape(-1),
                                   reduction="sum").item()
            n += seg.shape[1] - 1
    p_qat = float(np.exp(tot / n))
    print(f"\n=== QAT 后 ppl = {p_qat:.2f} ===")
    print(f"  fp16 {p_fp:.2f} → 量化 {p_q:.2f} → QAT {p_qat:.2f}"
          f"  （恢复 {(p_q-p_qat)/(p_q-p_fp)*100:.0f}% 的损伤）")
    json.dump(dict(p_fp=p_fp, p_quant=p_q, p_qat=p_qat, kappa=args.kappa,
                   layers=args.train_layers, steps=args.steps),
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dz",
                                f"qat_{args.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
