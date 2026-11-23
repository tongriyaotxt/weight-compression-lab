"""paths.py — 所有脚本共用的路径解析。

本机路径不写进代码。默认值相对仓库根目录，可用环境变量覆盖：

    QWEN8B_IR   OpenVINO IR 目录（Qwen3-8B INT4）
    QWEN17B_IR  OpenVINO IR 目录（Qwen3-1.7B INT4）
    LAB_OUT     产物目录（默认 labs/dz）
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

IR8B = os.environ.get("QWEN8B_IR", os.path.join(ROOT, "models", "npu-qwen3-8b"))
IR17B = os.environ.get("QWEN17B_IR", os.path.join(ROOT, "models", "npu-qwen3-1.7b"))
OUTDIR = os.environ.get("LAB_OUT", os.path.join(HERE, "dz"))
RES = os.path.join(ROOT, "results")
