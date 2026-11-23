"""dedup.py — 查模型里有没有整块重复的张量（embedding tying 是最常见的一种）。"""
import hashlib
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

fh = open(os.path.join(IR, "openvino_model.bin"), "rb")
mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)

print("=== 所有 u8 张量 ===")
for nm, c in C.items():
    if c["et"] == "u8":
        print(f"  {c['size']/2**20:8.1f} MiB  {c['shape']}  {nm}")

print("\n=== 逐字节比对 embed_tokens vs lm_head ===")
a = C["self.model.embed_tokens.weight"]
b = C["self.lm_head.weight"]
print(f"  a: {a['shape']} @ {a['off']}  {a['size']}")
print(f"  b: {b['shape']} @ {b['off']}  {b['size']}")
n = min(a["size"], b["size"])
same = 0
CH = 8 << 20
for p in range(0, n, CH):
    q = min(p + CH, n)
    ba = mm[a["off"] + p: a["off"] + q]
    bb = mm[b["off"] + p: b["off"] + q]
    same += sum(x == y for x, y in zip(ba, bb)) if False else int(np.frombuffer(ba, np.uint8).tolist() == np.frombuffer(bb, np.uint8).tolist()) * (q - p)
print(f"  相同字节: {same}/{n} = {100*same/n:.2f}%")

print("\n=== 前 64 字节对比 ===")
print("  embed :", mm[a["off"]:a["off"] + 64].hex())
print("  lm_head:", mm[b["off"]:b["off"] + 64].hex())

print("\n=== 全模型哈希（找完全相同的块）===")
seen = {}
dups = []
for nm, c in sorted(C.items()):
    if c["size"] < 1 << 20:
        continue
    if c["et"] not in ("i4", "u8", "f16"):
        continue
    h = hashlib.blake2b(mm[c["off"]:c["off"] + c["size"]], digest_size=16).hexdigest()
    if h in seen:
        dups.append((seen[h], nm, c["size"]))
    else:
        seen[h] = nm
if dups:
    for x, y, s in dups:
        print(f"  重复: {x}  ==  {y}   ({s/2**20:.1f} MiB)")
    print(f"  可省: {sum(s for _, _, s in dups)/2**20:.1f} MiB")
else:
    print("  没有完全相同的块")
