# Weight Compression Lab

**English | [中文](README.zh-CN.md)**

Forensics on what is actually compressible in LLM weights — and compression only where the evidence says it exists.

A reproducible experiment repo. It opens the OpenVINO INT4 IR of Qwen3-8B/1.7B, measures the true entropy of the weights, falsifies every plausible source of structural redundancy one by one, entropy-codes what remains, then checks on a real model whether the loss in quality is real, and finally tests on small models whether "writing the bit rate into the training objective" actually holds up.

**One-line conclusion:** trained weights are i.i.d. — all compressibility lives in the marginal distribution. Lossless coding gets 22.1% at zero quality cost. Anything below that requires changing the weights themselves (rate-QAT or VQ); a better entropy coder will not get you there.

---

## Numbers

| Item | Result |
|---|---|
| Qwen3-8B INT4, full-model entropy coding | 4.718 GB → **3.677 GB (−22.1%, lossless)** |
| Distance from the entropy bound | **+0.03%** (static rANS, 12-bit tables; canonical Huffman is +1.50%) |
| Structural redundancy (5 tests / 4 scales) | **all land on the null hypothesis**: weights are i.i.d. |
| Distance from the R(D) bound | +0.206 bit (7.3%) = the 1.53 dB space-filling loss of scalar quantization |
| Qwen3-1.7B dequantized to torch, ppl | **18.466** (311/311 tensors) |
| 1.7B PTQ: 3.06 bpw is an exact identity | 2.44 bpw costs +114% ppl; 2.23 bpw is dead |
| Embedding at 6 bit | −28% size, ppl **unchanged** (−0.6%) |
| Embedding at 4 bit | −58% size, ppl +6% |
| Rate control (CBR) | target bpw reproduced to **±0.001 bit** across seeds |
| Dead zone at 1.0–1.5 bpw | worth **9%–22% of the rate** |

---

## What is in here

```
labs/                      experiments (each runs standalone)
  paths.py                 path resolution (env-var overridable, see below)
  ir_probe.py              unpack OpenVINO IR → numpy
  weight_forensics.py      H0/H1, tile self-similarity, low rank
  forensics2-5.py          follow-up probes (incl. the i4 two's-complement fix)
  cross.py / alloc.py      cross-layer correlation / reverse water-filling
  dedup.py / embed_link.py is lm_head a reuse of embed?
  codec.py / analysis3.py  rANS vs Huffman + codestream autocorrelation
  rd_headroom.py           how far from the R(D) bound we still are
  repack.py / repack_rans.py / repack_rans_write.py   QEC v1 (Huffman) / v2 (rANS)
  qec_block.py             table granularity: per-tensor vs per-block
  vocab_probe.py           lossless options for the vocabulary side
  shape.py / shape2.py     distribution shape → entropy at fixed distortion
  dz_train.py              dead-zone quantizer + rate objective + CBR controller
  dz_analyze.py / dz_prune.py / dz_calib.py   frontier / post-hoc downgrade / calibration
  ecsq_theory.py           analytic dz*(H) curve for ECSQ
  entropy_probe.py / middle_out_probe.py      theory-layer calibration
  real_qwen.py             real model: IR → torch Qwen3 (tokenizer + ppl)
  real_quant_curve.py      real-model PTQ curve (grid-aligned)
  real_sensitivity.py      per-layer sensitivity + the compounding law
  real_vocab.py            vocabulary side at reduced precision
  qat_real.py              real-model QAT (fp32 masters + layer freezing)
results/                   small result artifacts (csv/json)
docs/FINDINGS.md           every measured number, with mechanisms and pitfalls
```

---

## Setup

```bash
python 3.11+
torch >= 2.4      # CUDA build: pip install torch --index-url https://download.pytorch.org/whl/cu128
numpy
```

No model weights are included. Scripts read from `models/` by default; override with env vars:

```powershell
$env:QWEN8B_IR  = "<IR root>\npu-qwen3-8b"      # OpenVINO IR dir (Qwen3-8B INT4)
$env:QWEN17B_IR = "<IR root>\npu-qwen3-1.7b"    # same, 1.7B
$env:LAB_OUT    = "labs\dz"                     # training artifacts
```

An IR directory needs `openvino_model.xml` + `openvino_model.bin`; the real-model experiments also need `tokenizer.json` and `config.json`.

---

## Reproduce

```powershell
$env:PYTHONIOENCODING='utf-8'

# 1) forensics: the weights are i.i.d.
python labs\ir_probe.py
python labs\weight_forensics.py
python labs\cross.py

# 2) entropy coding: 22.1%, rANS within 0.03% of the bound
python labs\codec.py
python labs\repack_rans.py            # analytic accounting + real encoder check
python labs\repack_rans_write.py      # writes the 3.677 GB file and reads it back

# 3) training side: CBR pins bpw to a target
python labs\dz_train.py --steps 1500 --mode wd --qat 1 --clipste 1 --cbr 2.5 --tag cbr25 --seed 0
python labs\dz_analyze.py

# 4) dead zone: theory vs measurement
python labs\ecsq_theory.py
python labs\dz_train.py --steps 1500 --mode wd --qat 1 --clipste 1 --cbr 1.2 --dzfix 1.5 --tag dz15 --seed 0

# 5) real model
python labs\real_qwen.py check
python labs\real_qwen.py ppl          # 18.466
python labs\real_quant_curve.py       # 3.06 bpw identity → 2.44 bpw cliff
python labs\real_vocab.py             # embedding bits vs ppl
```

---

## Main findings

1. **There is no structural redundancy.** Five independent tests on four scales all sit on the null hypothesis. Compressibility exists only in each weight's marginal distribution.

2. **Lossless gets 22.1%, and the coder is already at the bound.** Static rANS is within 0.03% of the entropy; what remains is the 1.53 dB space-filling loss of scalar quantization — only vector quantization can recover that 7%.

3. **Below 1.5 bpw you must switch to rANS.** Canonical Huffman has a hard 1 bit/symbol floor: it costs +14% on a ternary solution and **+152%** on a sparse-binary one.

4. **Rate should be controlled, not penalized.** A λ penalty is bistable and can kill the model outright. Projecting onto a target bpw instead reproduces the rate to ±0.001 bit across seeds.

5. **The dead zone is not a compression gain, it is low-rate reachability.** It is worth 9%/22%/20% of the rate at 1.5/1.2/1.0 bpw, and the optimal dz moves from 1.5 to 2.0 as compression deepens — matching the analytic ECSQ curve.

6. **Post-hoc downgrade does not work.** Pruning a 4-bit model down to ternary costs +5.5 bpc; retraining at that rate costs +0.06. The value of training at the target rate is 5.3 bpc.

7. **PTQ must start from the FP16 original.** The deployed grid is a cliff (3.06 bpw is an identity, 2.44 bpw doubles perplexity), so re-quantizing an already-INT4 artifact is a non-experiment.

8. **Damage compounds layer by layer.** Each layer alone at 1.8 bpw costs ±10%; all 28 together cost +2672%. That is why per-layer bit allocation has nothing to allocate.

Full tables, mechanisms and pitfalls: **[docs/FINDINGS.md](docs/FINDINGS.md)**.

---

## What this is not

- **Not state of the art.** The compression ratio is on par with tile-level rANS work (both sit at the entropy bound), but there is no GEMM-fused on-the-fly decompression here, no VQ codebook, and nothing works below 2 bpw.
- **Training-side conclusions come from 1.33M / 5M parameter toy models.** They are not extrapolated to LLM scale.
- **Real-model QAT is not solved.** 8/28 layers for 600 steps recovered 0.24% of the damage; it needs all layers, a long run and a real corpus.

---

## License

MIT
