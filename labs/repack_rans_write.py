"""
repack_rans_write.py — 真的把 Qwen3-8B 打成 rANS 码流文件，再回读验证。

输出：
  labs/qec/model.qec.rans.bin   码流（每张量一段）
  labs/qec/model.qec.rans.json  索引（name/shape/et/offset/bytes/频率表）

格式 QEC v2：
  header : 每张量 { name, shape, et, offset, bytes, freq[16 或 256] }
  body   : 每张量一段 rANS，紧接排列

验证：随机抽 4 个张量，解码后与原始码流逐符号比对。
"""

import json
import mmap
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rans import build_table, decode, encode, encode_chunk  # noqa: E402

from paths import IR8B as IR
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qec")
os.makedirs(OUT, exist_ok=True)
BIN = os.path.join(OUT, "model.qec.rans.bin")
IDX = os.path.join(OUT, "model.qec.rans.json")


def load_index():
    root = ET.parse(os.path.join(IR, "openvino_model.xml")).getroot()
    C = {}
    for layer in root.iter("layer"):
        if layer.get("type") != "Const":
            continue
        d = layer.find("data")
        if d is None:
            continue
        nm = layer.get("name") or ""
        if not nm.endswith(".weight"):
            continue
        k = [x for x in (d.get("shape") or "").split(",") if x.strip()]
        C[nm] = dict(shape=tuple(int(x) for x in k), et=d.get("element_type"),
                     off=int(d.get("offset")), size=int(d.get("size")))
    return C


def read_codes(mm, c):
    r, cc = c["shape"]
    n = r * cc
    if c["et"] == "i4":
        raw = np.frombuffer(mm, np.uint8, n // 2, c["off"])
        p = np.empty(n, np.uint8)
        p[0::2] = raw & 0x0F
        p[1::2] = raw >> 4
        return p
    return np.frombuffer(mm, np.uint8, n, c["off"])


def main():
    C = load_index()
    fh = open(os.path.join(IR, "openvino_model.bin"), "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)

    idx = []
    t0 = time.time()
    written = 0
    with open(BIN, "wb") as out:
        for i, (nm, c) in enumerate(sorted(C.items())):
            sym = read_codes(mm, c)
            nsym = 16 if c["et"] == "i4" else 256
            cnt = np.bincount(sym, minlength=nsym)
            f, cum = build_table(cnt.astype(np.float64), nsym=nsym)
            CH = 2_000_000
            state, nb = None, 0
            for i0 in range(0, len(sym), CH):
                blob, state = encode_chunk(sym[i0:i0 + CH].tolist(), f, cum, state)
                out.write(blob)
                nb += len(blob)
            out.write(int(state).to_bytes(4, "little"))
            idx.append(dict(name=nm, shape=list(c["shape"]), et=c["et"],
                            offset=written, bytes=nb, state=int(state),
                            freq=f.tolist()))
            written += nb + 4
            if i % 20 == 0:
                print(f"  [{i:3d}/254] {nm[:52]:<52} {nb/1e6:7.2f} MB"
                      f"  累计 {written/1e9:.3f} GB  {time.time()-t0:.0f}s", flush=True)
            del sym, blob

    json.dump(dict(format="QEC v2 (static rANS, 12-bit)", source=IR,
                   bytes=written, tensors=idx), open(IDX, "w"))
    print(f"\n写完：{written/1e9:.3f} GB  用 {time.time()-t0:.0f}s")
    print(f"文件 {BIN}  ({os.path.getsize(BIN)/1e9:.3f} GB)")

    # 回读验证
    print("\n回读验证（随机 4 个张量）")
    fh2 = open(BIN, "rb")
    mm2 = mmap.mmap(fh2.fileno(), 0, access=mmap.ACCESS_READ)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(idx), 4, replace=False)
    allok = True
    for p in pick:
        e = idx[p]
        nsym = 16 if e["et"] == "i4" else 256
        f = np.array(e["freq"], np.int64)
        cum = np.concatenate([[0], np.cumsum(f)[:-1]])
        blob = mm2[e["offset"]:e["offset"] + e["bytes"]]
        state = int.from_bytes(mm2[e["offset"] + e["bytes"]:e["offset"] + e["bytes"] + 4], "little")
        n = int(np.prod(e["shape"]))
        back = decode(blob, state, f, cum, n)
        orig = read_codes(mm, C[e["name"]])
        ok = np.array_equal(orig, back)
        allok &= ok
        print(f"  {e['name'][:50]:<50} {n/1e6:7.2f}M 符号  {'OK' if ok else 'FAIL'}")
    print(f"\n全部通过：{allok}")


if __name__ == "__main__":
    main()
