"""diag.py — 布局诊断：确认 i4 张量到底怎么存的。"""
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
                                off=int(d.get("offset")), size=int(d.get("size")))

print("文件大小      :", os.path.getsize(os.path.join(IR, "openvino_model.bin")))
print("Σ size        :", sum(v["size"] for v in C.values()))
print("最大 offset+size:", max(v["off"] + v["size"] for v in C.values()))

fh = open(os.path.join(IR, "openvino_model.bin"), "rb")
mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)

nm = "self.model.layers.18.mlp.up_proj.weight"
c = C[nm]
print(f"\n{nm}  {c['shape']}  et={c['et']}  off={c['off']} size={c['size']}")
print("前 32 字节:", mm[c["off"]:c["off"] + 32].hex())

raw = np.frombuffer(mm, np.uint8, c["size"], c["off"])
lo = raw & 0x0F
hi = raw >> 4
print("\n低 nibble H0:", end=" ")
for name, arr in (("lo", lo), ("hi", hi), ("all", np.concatenate([lo, hi]))):
    cnt = np.bincount(arr, minlength=16)
    print(f"{name}: " + " ".join(f"{x/len(arr)*100:.1f}" for x in cnt))

# 假设 1：低 nibble 属于前半张量、高 nibble 属于后半张量
n = int(np.prod(c["shape"]))
p = np.empty(2 * len(raw), dtype=np.uint8)
p[0::2] = lo
p[1::2] = hi
W = p.reshape(c["shape"])
row = W[0]
print(f"\n单行(4096) 直方图: " + " ".join(f"{x}" for x in np.bincount(row, minlength=16)))
print(f"单行 均值 {row.mean():.2f} 标准差 {row.std():.2f} 唯一值 {len(np.unique(row))}")

# 假设 2：两个 nibble 分别属于两半
half = len(raw)
A = lo.reshape(-1)
B = hi.reshape(-1)
print(f"\n前半 lo 均值 {A.mean():.2f} 后半 hi 均值 {B.mean():.2f}")

# 对照：u8 embedding 单行
nm2 = "self.model.embed_tokens.weight"
c2 = C[nm2]
e = np.frombuffer(mm, np.uint8, int(np.prod(c2["shape"])), c2["off"]).reshape(c2["shape"])
print(f"\nembedding 单行 均值 {e[0].mean():.1f} 标准差 {e[0].std():.1f} "
      f"min {e[0].min()} max {e[0].max()}")
print("embedding 前 32 值:", e[0][:32].tolist())
