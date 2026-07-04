# PROGRESS — Jormungandr

Detailed experiment status and reproduction notes. See `docs/stage2.md` for the
distillation pipeline and `MODEL_CARD.md` for public model documentation.
**Last updated: 2026-07-04.**

Headline model: Stage 2 EQTransformer-distilled student (epoch 20, val weighted
BCE 0.017934). The reproducible training checkpoint path is
`checkpoints/stage2_distill/best.pt`; the repository ships FP32/INT8 ONNX
artifacts instead of PyTorch checkpoints. Same 48,051-param architecture as
Stage 1 — distillation only changed the weights.

---

## 1. Phase status + headline metrics

| Phase | Scope | Status |
|---|---|---|
| 1 | data pipeline (preprocess, labels, augment, grouped split) | ✅ complete & verified on STEAD |
| 2 | 1D U-Net (48,051 params, 38.1 MFLOPs, INT8-friendly ops) | ✅ complete & verified |
| 3 | training — Stage 1 supervised + Stage 2 distillation | ✅ complete |
| 4 | evaluation — F1 / pick MAE / SNR buckets / EQT baseline / threshold tuning | ✅ complete |
| 5 | deployment — ONNX → INT8 → latency → streaming | ✅ complete |

**Headline test-split numbers** (7,781 traces: 4,957 eq / 2,824 noise; identical
split + tolerances across all rows):

| model / operating point | params | P | R | **F1** | FP noise | FN eq | P MAE±std | S MAE±std |
|---|---|---|---|---|---|---|---|---|
| **Stage 2 distilled — max-F1 (thr 0.80)** | 48,051 | 0.9910 | 0.9978 | **0.9944** | 45 | 11 | **36.9±56.3 ms** | **71.4±109.4 ms** |
| Stage 2 distilled — low-false-alarm (thr 0.90 + 500 ms) | 48,051 | 0.9977 | 0.9831 | 0.9903 | 11 | 84 | 36.9±56.3 ms | 71.4±109.4 ms |
| Stage 2 distilled — default (thr 0.50) | 48,051 | 0.9649 | 0.9994 | 0.9819 | 180 | 3 | 36.9±56.3 ms | 71.4±109.4 ms |
| Stage 1 supervised — max-F1 (thr 0.90 + 500 ms) | 48,051 | 0.9894 | 0.9964 | 0.9929 | 53 | 18 | 46.0±77.4 ms | 72.4±112.3 ms |
| Stage 1 supervised — default (thr 0.50) | 48,051 | 0.9227 | 0.9998 | 0.9597 | 415 | 1 | 46.0±77.4 ms | 72.4±112.3 ms |
| EQTransformer teacher (thr 0.50) | 376,935 | 0.993 | 0.979 | 0.9860 | 35 | 103 | 62.8±88.4 ms | 78.9±117.5 ms |
| EQTransformer teacher (best-F1 @ 0.15) | 376,935 | 0.984 | 0.990 | 0.9867 | 81 | 51 | 62.8±88.4 ms | 78.9±117.5 ms |

Distillation vs Stage 1 (same model): at thr 0.50 noise FP **415 → 180**, F1
**0.9597 → 0.9819**, P pick MAE **46.0 → 36.9 ms**. Tuned max-F1 **0.9929 → 0.9944**
(also beats the 7.8×-larger teacher's best, 0.9867). Val weighted BCE ~identical
between stages (0.01794 vs 0.01793) — the gains live in the detection/pick stream
shape, not the aggregate loss.

---

## 2. Current best operating point (recommended default)

**Stage 2 distilled, max-F1:** threshold **0.80**, min-duration **none**
(1 sample / 10 ms — distillation cleaned up noise, so no min-duration rule is
needed at the max-F1 point).

- F1 **0.9944** · precision 0.9910 · recall 0.9978
- FP on noise **45 / 2,824** · FN on earthquakes **11 / 4,957**
- P pick MAE **36.9 ms** (std 56.3) · S pick MAE **71.4 ms** (std 109.4)

Alternative — **low-false-alarm:** thr **0.90** + min-duration **500 ms** →
F1 0.9903, P 0.9977, R 0.9831, **FP 11**, FN 84 (0.4 % noise false-alarm rate).
Alternative — **recall-preserving:** thr 0.25 + 100 ms → R 0.9998 (FN 1), FP 418.

---

## 3. Exact commands + results

Prereqs: `cd ~/seismic-edge-picker && source .venv/bin/activate`. All eval/deploy
runs are CPU-capable; training used CUDA.

**Stage 1 — supervised training** (`checkpoints/stage1/best.pt`, epoch 48, val
weighted BCE 0.017936; 50 epochs, ~98 s/epoch, no overfitting):
```bash
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml \
    --checkpoint checkpoints/stage1/best.pt --out outputs/stage1_eval
```
→ test wBCE 0.01770; F1 0.9597 @0.50 (P 0.9227 / R 0.9998 / FP 415 / FN 1);
P MAE 46.0 ms, S MAE 72.4 ms. Artifacts in `outputs/stage1_eval/`.

**EQTransformer baseline** (pretrained SeisBench `stead`, same test traces +
tolerances, fed the project pipeline):
```bash
python scripts/eqtransformer_baseline.py --config configs/default.yaml \
    --out outputs/eqtransformer_baseline
```
→ 376,935 params, 1.51 MB, ~540 tr/s; F1 0.9860 @0.50 (P 0.993 / R 0.979 /
FP 35 / FN 103), best-F1 0.9867 @0.15. Artifacts in `outputs/eqtransformer_baseline/`.

**Stage 2 — distillation** (teacher cache → fine-tune from Stage 1;
`checkpoints/stage2_distill/best.pt`, epoch 20, val wBCE 0.017934; alpha 0.5,
temperature 1.0, non-augmented cache-aligned windows):
```bash
python scripts/cache_teacher.py --config configs/default.yaml                 # 2a: chunked/resumable EQT teacher cache → data/teacher_cache/
python scripts/train_distill.py --config configs/default.yaml \
    --init checkpoints/stage1/best.pt                                          # 2b: distillation fine-tune (val-loss checkpointing)
python scripts/evaluate.py --config configs/default.yaml \
    --checkpoint checkpoints/stage2_distill/best.pt --out outputs/stage2_eval
```
→ test wBCE 0.01771; F1 0.9819 @0.50 (P 0.9649 / R 0.9994 / FP 180 / FN 3);
P MAE 36.9 ms, S MAE 71.4 ms; ~640 tr/s. Artifacts in `outputs/stage2_eval/`.
Cheap smoke (24 traces + 1 epoch): `python scripts/distill_smoke.py --config configs/default.yaml`.

**Threshold + min-duration sweep** (postprocessing only, no retrain):
```bash
python scripts/threshold_sweep.py --config configs/default.yaml \
    --checkpoint checkpoints/stage2_distill/best.pt --out outputs/stage2_eval
```
→ operating points in §2 above; `threshold_sweep.csv/png`,
`threshold_recommendations.json` in `outputs/stage2_eval/`.

**ONNX export + parity** (Phase 5, done):
```bash
python scripts/export_onnx.py --config configs/default.yaml \
    --checkpoint checkpoints/stage2_distill/best.pt \
    --out outputs/onnx/stage2_distill.onnx
```
→ `outputs/onnx/stage2_distill.onnx` (opset 17, 0.20 MB, dynamic batch axis,
input `waveform` → output `streams`). Parity vs PyTorch (tol 1e-4, PASS):
dummy `(1,3,6000)` max abs err 1.107e-07 / mean 2.407e-08; real `(8,3,6000)`
max abs err 7.451e-07 / mean 3.308e-08. Report: `outputs/onnx/stage2_distill_parity.json`.

**INT8 static quantization + parity/eval** (Phase 5b, done):
```bash
python scripts/quantize_onnx.py --config configs/default.yaml \
    --checkpoint checkpoints/stage2_distill/best.pt \
    --fp32-onnx outputs/onnx/stage2_distill.onnx \
    --int8-out  outputs/onnx/stage2_distill_int8.onnx --threshold 0.80
```
ORT static quant (QDQ, per-channel weights QInt8, activations QUInt8, MinMax
calibration on **500 val traces**). Quantizes two variants — full INT8 and
body-INT8/head-FP32 (`/head/Conv` excluded) — evaluates both on the full test
split and ships the better. Detection metrics were identical; full INT8 had
marginally lower combined pick MAE, so **full** was shipped.
`outputs/onnx/stage2_distill_int8.onnx` **0.104 MB** vs FP32 0.204 MB →
**1.95× smaller (−48.8 %)**.

Full-test eval @ thr 0.80 (7,781 traces; FP32 ONNX row reproduces the headline
PyTorch numbers exactly, confirming the eval path):

| model | F1 | P | R | FP | FN | P MAE | S MAE |
|---|---|---|---|---|---|---|---|
| FP32 ONNX | **0.9944** | 0.9910 | 0.9978 | 45 | 11 | 37.0 ms | 71.4 ms |
| INT8 ONNX | **0.9920** | 0.9941 | 0.9899 | 29 | 50 | 45.6 ms | 76.2 ms |

Verdict: **no meaningful degradation** — F1 −0.0024 (< 0.005), P MAE +8.7 ms /
S MAE +4.8 ms (< 10 ms). INT8 trades recall for precision (FN 11→50, FP 45→29).
FP32↔INT8 parity is reported, not gated: dummy max 1.5e-2 / mean 9.3e-4, real
max 0.92 / mean 2.6e-2; the worst-case output error does not translate into a
large aggregate task-metric shift. Reports: `outputs/onnx/quantization_report.json`,
`outputs/stage2_int8_eval/int8_eval.json`.

**CPU latency benchmark** (Phase 5c, done):
```bash
python scripts/benchmark_latency.py --config configs/default.yaml \
    --checkpoint checkpoints/stage2_distill/best.pt \
    --fp32-onnx outputs/onnx/stage2_distill.onnx \
    --int8-onnx outputs/onnx/stage2_distill_int8.onnx \
    --out-dir outputs/latency
```
AMD EPYC 7763; input `(1,3,6000)`; one CPU thread; ORT sequential execution;
20 warmups excluded + 200 measured runs per backend:

| backend | p50 | p95 | mean | throughput |
|---|---:|---:|---:|---:|
| PyTorch CPU | 3.149 ms | 4.119 ms | 3.371 ms | 296.7 windows/s |
| FP32 ONNX Runtime | 1.543 ms | 1.571 ms | 1.567 ms | 638.0 windows/s |
| INT8 ONNX Runtime | **1.226 ms** | **1.253 ms** | **1.233 ms** | **811.0 windows/s** |

INT8 ONNX mean latency is **1.27× faster than FP32 ONNX** and **2.73× faster
than PyTorch** on this host. Reports: `outputs/latency/latency_report.json` and
`outputs/latency/latency_report.md`. CUDA was not benchmarked; deployment is CPU-only.

**Streaming inference** (Phase 5d, done):
```bash
python scripts/stream_infer.py --demo-traces 4 --plot --save-probabilities \
    --out-dir outputs/streaming_demo
```
Reusable core: `src/seismic_edge_picker/streaming.py`. INT8 ONNX defaults to
60 s windows / 30 s hop / one CPU thread; the padded tail is trimmed and overlap
probabilities are uniformly averaged. Events use threshold 0.80 + 10 ms, with
qualifying fragments up to 0.5 s apart coalesced. P/S peaks use 0.30 and are
emitted when associated with an event (1 s margin).

Smoke demo: four interleaved test traces (240 s; two earthquake / two noise),
seven windows → **2 events, 2 P picks, 1 S pick**. Outputs:
`events.csv`, `picks.csv`, `summary.{json,txt}`, optional
`merged_probabilities.npz`, and `streaming_predictions.png` under
`outputs/streaming_demo/`. Timestamps are relative to input start. This verifies
the path; it is not a new accuracy evaluation.

**INT8 low-false-alarm limitation:** 0.90 + 500 ms was selected on FP32 trace-level
outputs. It remains configurable, but has not been retuned or validated on merged
INT8 streaming output; the streaming default remains the evaluated 0.80 point.

---

## 4. Known caveats

- **EQT comparison is uncharitable to the teacher.** EQTransformer (baseline and
  the distillation teacher) was fed the project's demean+bandpass(1–45 Hz)+std
  pipeline, **not** its native preprocessing → its pick MAE is a *lower bound* on
  true capability; cross-model pick comparisons should not be over-read. Also:
  single fixed 60 s windows, no overlap/stacking (EQT's `classify()` normally
  stacks overlapping windows on continuous data).
- **Val loss ≈ unchanged Stage 1 → Stage 2** (0.01794 → 0.01793); the real gains
  are in detection precision and pick sharpness, visible only in the F1/MAE
  metrics, not the aggregate weighted BCE.
- **Stage 2 trains on deterministic, non-augmented, cache-aligned windows.**
  `distill.use_augmentations=false` is enforced and guarded — enabling it would
  desync inputs from the cached teacher soft targets (trainer aborts). Augmentation-
  aware distillation (online teacher, or augmenting the cached stream) is future
  work, not part of this run. See `docs/stage2.md`.
- **Teacher cache ~2.1 GB fp16, loaded fully into RAM** during distillation;
  switch to memmap if RAM-bound.
- **STEAD download is complete — do not re-trigger it.** Cached-only loading is on
  (`data.require_cached: true`). STEAD stores arrival samples as array-strings
  (e.g. `'[[5744.]]'`), parsed via `splits.parse_scalar` / `parse_snr_db`.
- **All inference/deployment code must be CPU-only** (Pi / Graviton targets); only
  training uses `train.device: cuda`.
- **Model is small (48,051 params)** — headroom to widen `model.encoder_channels`
  (still under the 300k budget) if accuracy plateaus.
- **ONNX export:** torch 2.12's default dynamo path needs `onnxscript`;
  `export_onnx.py` uses the legacy exporter (`dynamo=False`) to stay dependency-light.
- **Latency is hardware-specific.** Current numbers are from an AMD EPYC 7763
  host with one thread; rerun `benchmark_latency.py` on Raspberry Pi / Graviton.
- **The streaming demo is a path smoke test.** Concatenated, independently
  preprocessed traces have artificial boundaries and do not measure continuous-
  stream detection quality. Low-false-alarm INT8 streaming remains uncalibrated.

---

## 5. Phase 5 deployment checklist

1. **ONNX export** — ✅ done (`outputs/onnx/stage2_distill.onnx`, opset 17).
2. **Parity check** — ✅ done (max abs err 1.1e-7 dummy / 7.5e-7 real, tol 1e-4 PASS).
3. **INT8 static quantization** — ✅ done (`outputs/onnx/stage2_distill_int8.onnx`,
   0.104 MB, 1.95× smaller). Static QDQ, 500 val calibration traces; full-test F1
   0.9944 → 0.9920 (no meaningful loss). Head-FP32 fallback tried and found
   equivalent, so full INT8 shipped. `scripts/quantize_onnx.py`.
4. **Latency benchmark** — ✅ done (`scripts/benchmark_latency.py`). One thread,
   20 warmups + 200 runs on AMD EPYC 7763; INT8 ONNX p50/p95/mean =
   1.226/1.253/1.233 ms (811.0 windows/s). Rerun on Raspberry Pi / Graviton.
5. **Streaming wrapper** — ✅ done (`src/seismic_edge_picker/streaming.py`,
   `scripts/stream_infer.py`). 60 s / 30 s hop, overlap mean, relative event/P/S
   timestamps, CSV/JSON/text/plot outputs, four-trace demo, and synthetic tests.
   Default: 0.80 + 10 ms; INT8 low-false-alarm streaming is not yet calibrated.

---

## 6. Environment / verification

- Python 3.12, venv `.venv`. torch 2.12 (cu130, CUDA available), seisbench 0.11.7,
  numpy 2.5, scipy 1.18, matplotlib 3.11, pyyaml, pytest, **onnx 1.22.0 +
  onnxruntime 1.27.0** (CPU-only, Phase 5).
- STEAD at `~/.seisbench/datasets/stead/` (`metadata.csv` 402.6 MB +
  `waveforms.hdf5` 91.1 GB). Grouped split (subset 50k eq + 25k noise): train
  **59,116** / val **8,103** / test **7,781**, disjoint by event/station.
- `pytest -q` → **45 passed** (pure-logic tests; no dataset/network needed).
