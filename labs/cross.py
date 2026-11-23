"""
cross.py — 最后一块：跨层 / 跨投影的相似性。

层剪枝为什么能 work？如果深层高度冗余，那 W_{l+1} 应该和 W_l 高度相关，
于是可以存 Δ = W_{l+1} − W_l，残差方差远小于信号方差，直接省 bit。

用 R(D) 算收益：  省下的 bit = 0.5·log2(σ²/σ_Δ²)
"""

import mmap
import os
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
                                off=int(d.get("offset", 0)),
                                size=int(d.get("size")))
fh = open(os.path.join(IR, "openvino_model.bin"), "rb")
mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)


def deq(nm):
    c = C[nm]
    r, cc = c["shape"]
    n = r * cc
    raw = np.frombuffer(mm, np.uint8, n // 2, c["off"])
    p = np.empty(n, np.uint8)
    p[0::2] = raw & 0x0F
    p[1::2] = raw >> 4
    v = np.where(p > 7, p.astype(np.int16) - 16, p.astype(np.int16)).astype(np.float32)
    sc = C[nm + "/scale"]
    s = np.frombuffer(mm, np.float16, int(np.prod(sc["shape"])), sc["off"]).reshape(-1, 1)
    return v.reshape(r, cc) * s.astype(np.float32)


def stats(name, A, B):
    a = A.ravel()
    b = B.ravel()
    sa, sb = a.std(), b.std()
    corr = float(np.corrcoef(a, b)[0, 1])
    d = b - a
    sd = d.std()
    gain = 0.5 * np.log2(sb ** 2 / max(sd ** 2, 1e-30))
    print(f"  {name:<46} corr={corr:+.4f}  σ={sb:.5f}  σ_Δ={sd:.5f}  "
          f"省 {gain:+.3f} bit")
    return gain


print("=" * 92)
print("A  同类型投影，跨层")
print("=" * 92)
for base in ("mlp.up_proj", "mlp.down_proj", "self_attn.q_proj"):
    prev = None
    for L in (0, 6, 12, 18, 24, 30, 35):
        nm = f"self.model.layers.{L}.{base}.weight"
        if nm not in C:
            continue
        W = deq(nm)
        if prev is not None and prev[1].shape == W.shape:
            stats(f"{base}  L{prev[0]}→L{L}", prev[1], W)
        prev = (L, W)
    print()

print("=" * 92)
print("B  同一层内，不同投影之间")
print("=" * 92)
for L in (0, 18, 35):
    up = deq(f"self.model.layers.{L}.mlp.up_proj.weight")
    gate = deq(f"self.model.layers.{L}.mlp.gate_proj.weight")
    stats(f"L{L} up_proj vs gate_proj", up, gate)
    q = deq(f"self.model.layers.{L}.self_attn.q_proj.weight")
    o = deq(f"self.model.layers.{L}.self_attn.o_proj.weight")
    if q.shape == o.shape:
        stats(f"L{L} q_proj vs o_proj", q, o)
print()

print("=" * 92)
print("C  最远的两层（L0 vs L35）")
print("=" * 92)
a = deq("self.model.layers.0.mlp.up_proj.weight")
b = deq("self.model.layers.35.mlp.up_proj.weight")
stats("up_proj L0 vs L35", a, b)

print()
print("=" * 92)
print("D  结论用的阈值")
print("=" * 92)
print("  省 0.5 bit  = 需要 σ_Δ² / σ² < 0.71  (corr 大约 > 0.54)")
print("  省 1.0 bit  = 需要 σ_Δ² / σ² < 0.50  (corr 大约 > 0.71)")
print("  省 2.0 bit  = 需要 σ_Δ² / σ² < 0.25  (corr 大约 > 0.87)")
