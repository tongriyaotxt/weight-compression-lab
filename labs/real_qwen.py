"""
real_qwen.py — 把 OpenVINO IR 的 Qwen3-1.7B 反量化成 torch 模型。

背景（上一轮卡在这里）：`self.*` 命名的只有 7 个权重 + embed，RMSNorm 权重不见了。
本轮发现：**norm 的图节点带完整模块名**，例如
    __module.model.layers.0.input_layernorm/aten::mul/Multiply_1
它的第二个输入就是一个 shape=(1,1,2048) 的匿名 Constant —— 那就是 norm 权重。
所以按节点名做图追踪就能全部取回。

用法：
    python labs/real_qwen.py check     # 只检查张量齐不齐、统计量对不对
    python labs/real_qwen.py ppl       # 建模型 + 分词 + 算困惑度
"""

import json
import mmap
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from paths import IR17B as IR
CFG = json.load(open(os.path.join(IR, "config.json")))
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- IR 解析
class IRFile:
    def __init__(self, path=IR):
        self.dir = path
        self.root = ET.parse(os.path.join(path, "openvino_model.xml")).getroot()
        self.byid = {}
        self.consts = {}
        for layer in self.root.iter("layer"):
            self.byid[layer.get("id")] = layer
            if layer.get("type") != "Const":
                continue
            d = layer.find("data")
            if d is None:
                continue
            k = [x for x in (d.get("shape") or "").split(",") if x.strip()]
            self.consts[layer.get("name")] = dict(
                shape=tuple(int(x) for x in k) if k else (),
                et=d.get("element_type"), off=int(d.get("offset")), size=int(d.get("size")))
        self.fh = open(os.path.join(path, "openvino_model.bin"), "rb")
        self.mm = mmap.mmap(self.fh.fileno(), 0, access=mmap.ACCESS_READ)
        # 图边在 <edges> 里，不在 <layer> 里
        self.edge = {}
        for e in self.root.iter("edge"):
            self.edge[(e.get("to-layer"), e.get("to-port"))] = (e.get("from-layer"), e.get("from-port"))

    def raw_const(self, name):
        c = self.consts[name]
        dt = {"f32": np.float32, "f16": np.float16, "i4": np.uint8, "u8": np.uint8,
              "i32": np.int32, "i64": np.int64}[c["et"]]
        if c["et"] == "i4":
            n = int(np.prod(c["shape"]))
            raw = np.frombuffer(self.mm, np.uint8, n // 2, c["off"])
            p = np.empty(n, np.uint8)
            p[0::2] = raw & 0x0F
            p[1::2] = raw >> 4
            return p.reshape(c["shape"])
        return np.frombuffer(self.mm, dt, int(np.prod(c["shape"])), c["off"]).reshape(c["shape"])

    def weight(self, name):
        """i4/u8 张量 -> fp32，用同名 /scale（和 /zero_point）反量化。"""
        c = self.consts[name]
        code = self.raw_const(name).astype(np.float32)
        if c["et"] == "u8":
            zp = self.raw_const(name + "/zero_point").astype(np.float32)
            sc = self.raw_const(name + "/scale").astype(np.float32)
            return (code - zp.reshape(-1, 1)) * sc.reshape(-1, 1)
        if c["et"] == "i4":
            sc = self.raw_const(name + "/scale").astype(np.float32)
            # i4 码是二进制补码：code>=8 表示负数
            sgn = np.where(code > 7, code - 16, code)
            return sgn * sc.reshape(-1, 1)
        return code

    def norm(self, modname):
        """按模板块名找出 RMSNorm 权重：Multiply 的输入端口 0 由 Const 供给。"""
        for layer in self.root.iter("layer"):
            nm = layer.get("name") or ""
            if layer.get("type") != "Multiply" or not nm.startswith(modname + "/"):
                continue
            if not nm.endswith("Multiply_1"):
                continue
            for inp in layer.findall("input/port"):
                src = self.edge.get((layer.get("id"), inp.get("id")))
                if src is None:
                    continue
                node = self.byid.get(src[0])
                if node is None or node.get("type") != "Const":
                    continue
                c = self.consts.get(node.get("name"))
                if c and c["et"] in ("f32", "f16") and int(np.prod(c["shape"])) > 1:
                    return self.raw_const(node.get("name")).reshape(-1).astype(np.float32)
        return None


# ---------------------------------------------------------------- torch 模型
class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)) * self.w


def rope(x, pos, theta=1e6, dim=128):
    """标准 half-rotation RoPE。x: [B, H, T, D]"""
    half = dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, device=x.device, dtype=torch.float32) / half))
    t = pos.float()
    ang = torch.outer(t, freqs)
    cos, sin = ang.cos()[None, None].to(x.dtype), ang.sin()[None, None].to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Layer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, nh, nkv = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
        hd = cfg["head_dim"]
        self.nh, self.nkv, self.hd = nh, nkv, hd
        self.ln1 = RMSNorm(d, cfg["rms_norm_eps"])
        self.q = nn.Linear(d, nh * hd, bias=False)
        self.k = nn.Linear(d, nkv * hd, bias=False)
        self.v = nn.Linear(d, nkv * hd, bias=False)
        self.o = nn.Linear(nh * hd, d, bias=False)
        self.qn = RMSNorm(hd, cfg["rms_norm_eps"])
        self.kn = RMSNorm(hd, cfg["rms_norm_eps"])
        self.ln2 = RMSNorm(d, cfg["rms_norm_eps"])
        self.gate = nn.Linear(d, cfg["intermediate_size"], bias=False)
        self.up = nn.Linear(d, cfg["intermediate_size"], bias=False)
        self.down = nn.Linear(cfg["intermediate_size"], d, bias=False)

    def forward(self, x, pos):
        B, T, C = x.shape
        h = self.ln1(x)
        q = self.q(h).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k(h).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v(h).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q = self.qn(q)
        k = self.kn(k)
        q, k = rope(q, pos, dim=self.hd), rope(k, pos, dim=self.hd)
        k = k.repeat_interleave(self.nh // self.nkv, dim=1)
        v = v.repeat_interleave(self.nh // self.nkv, dim=1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).reshape(B, T, C)
        x = x + self.o(a)
        h2 = self.ln2(x)
        return x + self.down(F.silu(self.gate(h2)) * self.up(h2))


class Qwen3(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = nn.ModuleList([Layer(cfg) for _ in range(cfg["num_hidden_layers"])])
        self.norm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        x = self.emb(ids)
        for l in self.layers:
            x = l(x, pos)
        return self.head(self.norm(x))


# ---------------------------------------------------------------- 装载
def load(path=IR, dtype=torch.float32):
    ir = IRFile(path)
    m = Qwen3(CFG)
    P = m.state_dict()
    got = set()

    def put(name, arr):
        t = torch.from_numpy(np.ascontiguousarray(arr))
        if name not in P:
            raise KeyError(name)
        if tuple(P[name].shape) != tuple(t.shape):
            raise ValueError(f"{name}: {tuple(t.shape)} vs {tuple(P[name].shape)}")
        P[name].copy_(t.to(P[name].dtype))
        got.add(name)

    put("emb.weight", ir.weight("self.model.embed_tokens.weight"))
    if "self.lm_head.weight" in ir.consts:          # 8B 是解绑的
        put("head.weight", ir.weight("self.lm_head.weight"))
    else:                                            # 1.7B config: tie_word_embeddings=true
        put("head.weight", ir.weight("self.model.embed_tokens.weight"))
    fnorm = ir.norm("__module.model.norm")
    put("norm.w", fnorm)
    for i in range(CFG["num_hidden_layers"]):
        b = f"self.model.layers.{i}."
        m_ = f"__module.model.layers.{i}."
        put(f"layers.{i}.q.weight", ir.weight(f"{b}self_attn.q_proj.weight"))
        put(f"layers.{i}.k.weight", ir.weight(f"{b}self_attn.k_proj.weight"))
        put(f"layers.{i}.v.weight", ir.weight(f"{b}self_attn.v_proj.weight"))
        put(f"layers.{i}.o.weight", ir.weight(f"{b}self_attn.o_proj.weight"))
        put(f"layers.{i}.gate.weight", ir.weight(f"{b}mlp.gate_proj.weight"))
        put(f"layers.{i}.up.weight", ir.weight(f"{b}mlp.up_proj.weight"))
        put(f"layers.{i}.down.weight", ir.weight(f"{b}mlp.down_proj.weight"))
        for nm, key in (("input_layernorm", "ln1"), ("post_attention_layernorm", "ln2"),
                        ("self_attn.q_norm", "qn"), ("self_attn.k_norm", "kn")):
            w = ir.norm(f"{m_}{nm}")
            if w is None:
                raise KeyError(f"{m_}{nm}")
            put(f"layers.{i}.{key}.w", w)

    m.load_state_dict(P)
    return m.to(dtype).eval(), ir, got


def check():
    m, ir, got = load()
    n = sum(p.numel() for p in m.parameters())
    named = sum(1 for k in m.state_dict())
    print(f"装载完成：{n/1e9:.4f}B 参数，{len(got)}/{named} 个张量")
    miss = set(m.state_dict()) - got
    print("缺失：", miss if miss else "无")
    print("\n统计抽查（应看到 norm≈1 附近、权重 σ 合理）")
    for k in ("emb.weight", "layers.0.q.weight", "layers.0.down.weight",
              "layers.0.ln1.w", "layers.0.qn.w", "norm.w"):
        t = dict(m.named_parameters()).get(k)
        if t is None:
            t = m.emb.weight if k == "emb.weight" else None
        if t is not None:
            print(f"  {k:<22} shape={tuple(t.shape)}  mean={t.mean().item():+.4f}"
                  f"  std={t.std().item():.4f}  absmax={t.abs().max().item():.3f}")


# ---------------------------------------------------------------- 分词（byte-level BPE）
import re as _re  # noqa: E402


def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class Tok:
    def __init__(self, path=IR):
        j = json.load(open(os.path.join(path, "tokenizer.json"), encoding="utf-8"))
        self.vocab = j["model"]["vocab"]
        self.merges = {}
        for i, m in enumerate(j["model"]["merges"]):
            a, b = (m.split(" ") if isinstance(m, str) else m)
            self.merges[(a, b)] = i
        self.b2u = _bytes_to_unicode()
        # re 不支持 \p{L}（那要 regex 包）；ASCII 子集够做 ppl 验收
        self.pat = _re.compile(
            r"'(?:[sdmt]|ll|ve|re)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")

    def _bpe(self, word):
        syms = list(word)
        while len(syms) > 1:
            best, bi = None, None
            for i in range(len(syms) - 1):
                r = self.merges.get((syms[i], syms[i + 1]))
                if r is not None and (bi is None or r < bi):
                    bi, best = r, i
            if best is None:
                break
            syms[best:best + 2] = [syms[best] + syms[best + 1]]
        return syms

    def encode(self, text):
        ids = []
        for chunk in self.pat.findall(text):
            w = "".join(self.b2u[b] for b in chunk.encode("utf-8"))
            for s in self._bpe(w):
                ids.append(self.vocab[s])
        return ids


SAMPLE = """Compression is the art of removing what does not need to be said twice.
A language model is, in a very literal sense, a lossy compressor of its training
corpus: it learns to predict the next token, and in doing so it must internalise
the regularities of the data. The weights are the residue of that process. If the
regularities were all that mattered, the weights would be small; the fact that they
are large suggests that a great deal of the cost is spent on the particular path the
optimiser took rather than on the task itself. Rate-distortion theory gives us the
vocabulary to talk about this trade-off precisely, but it says nothing about which
directions in weight space are cheap to discard. That is an empirical question, and
the answer so far is that almost all of the redundancy lives in the marginal
distribution of each weight rather than in the correlations between them.
"""


def ppl(m, tok, text=SAMPLE, ctx=256, dtype=torch.bfloat16):
    ids = tok.encode(text)
    if len(ids) < ctx + 1:
        ids = ids * (2 + (ctx // max(len(ids), 1)))
    x = torch.tensor([ids], device=DEV)
    m = m.to(DEV).to(dtype)
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, x.shape[1] - 1, ctx):
            seg = x[:, i:i + ctx + 1]
            if seg.shape[1] < 3:
                break
            logits = m(seg[:, :-1]).float()
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   seg[:, 1:].reshape(-1), reduction="sum")
            tot += loss.item()
            n += seg.shape[1] - 1
    import math
    return math.exp(tot / n), n


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "check":
        check()
    elif cmd == "ppl":
        t0 = __import__("time").time()
        m, ir, got = load()
        tok = Tok()
        ids = tok.encode(SAMPLE)
        print(f"分词自检：{len(ids)} 个 token，前 12 个 = {ids[:12]}")
        print("  解码回文本：", repr(tok.__class__.__name__), "（无 decoder，略）")
        p, n = ppl(m, tok)
        print(f"Qwen3-1.7B（INT4 反量化，bf16 前向）  ppl = {p:.3f}   有效 token = {n}"
              f"   用时 {__import__('time').time()-t0:.0f}s")
