"""ir_probe.py — 直接解析 OpenVINO IR，把权重张量抠成 numpy 数组。

IR v11 结构：openvino_model.xml 里有 <layer type="Const">，
每个下面挂一个 <data offset= size= shape= element_type=/>，
offset 指向 openvino_model.bin 里的字节偏移。所以可以直接 mmap 读。
"""
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np

from paths import IR8B as IR_DIR
XML = os.path.join(IR_DIR, "openvino_model.xml")
BIN = os.path.join(IR_DIR, "openvino_model.bin")

NP = {"f16": np.float16, "f32": np.float32, "f64": np.float64,
      "i8": np.int8, "u8": np.uint8, "i4": None, "u4": None,
      "i32": np.int32, "i64": np.int64, "boolean": np.bool_}


def parse(xml_path):
    root = ET.parse(xml_path).getroot()
    out = []
    for layer in root.iter("layer"):
        if layer.get("type") != "Const":
            continue
        d = layer.find("data")
        if d is None:
            continue
        out.append({
            "name": layer.get("name"),
            "shape": d.get("shape"),
            "et": d.get("element_type"),
            "offset": int(d.get("offset", -1)),
            "size": int(d.get("size", -1)),
        })
    return out


def load(const, mm):
    et = const["et"]
    shape = tuple(int(x) for x in const["shape"].split(","))
    n = int(np.prod(shape))
    off = const["offset"]
    dt = NP.get(et)
    if dt is None:
        return None, shape
    raw = np.frombuffer(mm, dtype=dt, count=n, offset=off)
    return raw.reshape(shape), shape


def main():
    consts = parse(XML)
    print(f"consts: {len(consts)}")
    print("element_type 分布:", Counter(c["et"] for c in consts).most_common())
    print()

    big = sorted(consts, key=lambda c: -c["size"])
    print(f"{'size(MB)':>10}{'shape':>22}{'et':>8}  name")
    print("-" * 100)
    for c in big[:25]:
        print(f"{c['size']/2**20:>10.2f}{c['shape']:>22}{c['et']:>8}  {c['name']}")

    # 二维大张量 = 矩阵乘权重
    mats = [c for c in consts if len(c["shape"].split(",")) == 2 and c["size"] > 2**20]
    print(f"\n二维大矩阵: {len(mats)} 个")
    with open(BIN, "rb") as f:
        mm = memoryview(f.read())
    for c in mats[:3]:
        a, shape = load(c, mm)
        print(f"\n{c['name']}  shape={shape} et={c['et']}")
        if a is not None:
            a = a.astype(np.float32)
            print(f"  mean={a.mean():+.5f} std={a.std():.5f} "
                  f"min={a.min():+.4f} max={a.max():+.4f} "
                  f"kurtosis={((a-a.mean())**4).mean()/a.std()**4:.2f}")


if __name__ == "__main__":
    main()
