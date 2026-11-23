**[English](README.md) | 中文**

# Weight Compression Lab

对 LLM 权重的**可压性做取证，然后只压能压的部分**。

一个可复现的实验仓库：从 OpenVINO INT4 的 IR 里把权重挖出来，量它的真实熵、
把结构冗余逐条证伪、做熵编码、在真模型上量困惑度、并在小模型上验证
"把码率写进训练目标"到底成不成立。

**一句话结论**：训练出来的权重是 i.i.d. 的，可压性全在边际分布里；
无损能拿 22.1%（零损失），再往下必须改权重本身（rate-QAT / VQ），不能靠更好的编码器。

---

## 数字速览

| 项目 | 结果 |
|---|---|
| Qwen3-8B INT4 全模型熵编码 | 4.718 GB → **3.677 GB（−22.1%，零损失）** |
| 编码器距熵界 | **+0.03%**（rANS，12-bit 表；Huffman 是 +1.50%） |
| 结构冗余（5 类检验 / 4 个尺度） | **全部为零假设**：权重 = i.i.d. |
| 距 R(D) 界 | +0.206 bit（7.3%）= 标量量化的 1.53 dB 格填充损失 |
| Qwen3-1.7B 反量化 → torch，ppl | **18.466**（311/311 张量） |
| 1.7B PTQ：3.06 bpw 恒等 | 2.44 bpw 就 +114% ppl，2.23 bpw 直接死 |
| embedding 6 bit | 体积 −28%，ppl **不降**（−0.6%） |
| embedding 4 bit | 体积 −58%，ppl +6% |
| 码率控制（CBR） | 目标 bpw 跨 seed 误差 **±0.001 bit** |
| 死区（dz）在 1.0–1.5 bpw 的收益 | **9%–22% 的码率** |

---

## 仓库结构

```
labs/                      实验脚本（全部可独立运行）
  paths.py                 路径解析（环境变量覆盖，见下）
  ir_probe.py              OpenVINO IR 解包 → numpy
  weight_forensics.py      H0/H1、tile 自相似、低秩
  forensics2-5.py          迭代取证（含 i4 补码修正）
  cross.py / alloc.py      跨层相关 / 反向注水
  dedup.py / embed_link.py embed 与 lm_head 的关系
  codec.py / analysis3.py  rANS vs Huffman + 码流自相关
  rd_headroom.py           距 R(D) 界还剩多少
  repack.py / repack_rans.py / repack_rans_write.py   QEC v1(Huffman) / v2(rANS) 打包
  qec_block.py             表粒度（per-tensor vs per-block）
  vocab_probe.py           词表侧无损手段
  shape.py / shape2.py     分布形状 → 固定失真下的熵
  dz_train.py              死区量化器 + 码率目标 + CBR 控制器（训练侧核心）
  dz_analyze.py / dz_prune.py / dz_calib.py   前沿 / 后验降档 / 标定
  ecsq_theory.py           ECSQ 的 dz*(H) 解析曲线
  entropy_probe.py / middle_out_probe.py      理论层校准
  real_qwen.py             真模型：IR → torch Qwen3（含分词与 ppl）
  real_quant_curve.py      真模型 PTQ 曲线（网格对齐）
  real_sensitivity.py      逐层敏感性 + 连乘律
  real_vocab.py            词表侧降精度
  qat_real.py              真模型 QAT（fp32 主权重 + 分层冻结）
results/                   小体量结果（csv/json）
docs/FINDINGS.md           全部实测结论与技术细节
```

---

## 环境

```bash
python 3.11+
torch >= 2.4      # CUDA 版：pip install torch --index-url https://download.pytorch.org/whl/cu128
numpy
```

本仓库不包含模型权重。默认从 `models/` 读取，可用环境变量覆盖：

```powershell
$env:QWEN8B_IR  = "<IR 根目录>\npu-qwen3-8b"      # OpenVINO IR 目录（Qwen3-8B INT4）
$env:QWEN17B_IR = "<IR 根目录>\npu-qwen3-1.7b"    # 同上，1.7B
$env:LAB_OUT    = "labs\dz"                     # 训练产物目录
```

IR 目录需要 `openvino_model.xml` + `openvino_model.bin`；
真模型实验还需要 `tokenizer.json` 与 `config.json`。

---

## 复现

```powershell
$env:PYTHONIOENCODING='utf-8'

# 1) 取证：权重是 i.i.d. 的
python labs\ir_probe.py
python labs\weight_forensics.py
python labs\cross.py

# 2) 熵编码：22.1%，rANS 距熵界 0.03%
python labs\codec.py
python labs\repack_rans.py            # 解析账 + 真实编码验证
python labs\repack_rans_write.py      # 写 3.677 GB 文件并回读

# 3) 训练侧：CBR 把 bpw 钉到目标值
python labs\dz_train.py --steps 1500 --mode wd --qat 1 --clipste 1 --cbr 2.5 --tag cbr25 --seed 0
python labs\dz_analyze.py

# 4) 死区理论 vs 实测
python labs\ecsq_theory.py
python labs\dz_train.py --steps 1500 --mode wd --qat 1 --clipste 1 --cbr 1.2 --dzfix 1.5 --tag dz15 --seed 0

# 5) 真模型
python labs\real_qwen.py check
python labs\real_qwen.py ppl          # 18.466
python labs\real_quant_curve.py       # 3.06 bpw 恒等 → 2.44 bpw 悬崖
python labs\real_vocab.py             # embedding bits vs ppl
```

---

## 主要结论

1. **权重没有结构冗余。** 五种独立检验、四个尺度，全部落在零假设上。
   可压性只存在于每个权重的边际分布里。
2. **无损能拿 22.1%，且编码器已经贴界。** 静态 rANS 距熵界 0.03%，
   再往下是标量量化的 1.53 dB 格填充损失——只有矢量量化能拿回那 7%。
3. **子 1.5 bpw 必须换 rANS。** Huffman 有 1 bit/符号的硬地板：
   在三元解上贵 14%，在稀疏二元解上贵 **152%**。
4. **码率该被控制，不该被惩罚。** 用 λ 会双稳态甚至压死模型；
   投影到目标 bpw 后跨 seed 误差 ±0.001 bit。
5. **死区不是压缩增益，是低码率可达性。** 1.5/1.2/1.0 bpw 分别值 9%/22%/20% 的码率，
   且最优 dz 随压缩加深从 1.5 移到 2.0。
6. **后验降档不可行。** 4-bit 模型剪到三元 +5.5 bpc；在目标码率下重训只花 +0.06。
7. **PTQ 只能在 FP16 原件上做。** 部署网格是悬崖（3.06 bpw 恒等，2.44 bpw 翻倍），
   在已量化的 INT4 产物上再量化是伪命题。
8. **损伤按层连乘。** 每层单独 1.8 bpw 只 ±10%，28 层一起 +2672% —— 逐层分配没有着力点。

详细的表格、机制与踩坑记录见 **[docs/FINDINGS.md](docs/FINDINGS.md)**。

---

## 边界（明确说清楚）

- **不是 SOTA。** 压缩比与 ISCA'26 的 tile-rANS 同级（都贴熵界），
  但本仓库没有 GEMM 融合的在线解压，也没有 VQ 码本，2 bpw 以下无解。
- **训练侧结论来自 1.33M / 5M 参数的 toy 模型**，未在 LLM 规模验证。
- **真模型 QAT 未跑通**：8/28 层 600 步只恢复 0.24%，需要全层 + 长训 + 大语料。

---

## License

MIT
