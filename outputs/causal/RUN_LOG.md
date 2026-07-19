# Causal early-firing variant — run log

Branch: `feature/causal-model`.

## Stage 1 — causal architecture
- Committed `b8afd35`: `SeismicUNet(causal=True, lookahead=L)` using left-only Conv1d padding and causal decoder alignment.
- Decision: causal/lookahead flag on `SeismicUNet`, not a separate model file, because parameter shapes stay identical for warm-start.
- Verified by `tests/test_causal.py`: no future leakage for `t < i-L`; acausal model leaks as expected; acausal checkpoint loads into causal shapes.

## Stage 2 — causal preprocessing
- Committed `4ae1cac`: forward-only Butterworth filtering, causal running normalization, and stateful `CausalPreprocessor` for chunked use.
- Warm-start uses real leading/background samples for filter state; no zero-filled streaming warmup.
- Verified by `tests/test_causal_preprocessing.py`: future perturbations do not alter earlier preprocessing output; chunked state matches one-shot causal preprocessing.

## Stage 3 — retraining plumbing [DATASET-GATED]
- Committed `f33b0b5`: `scripts/train_distill.py --causal --data` reuses the existing distillation loss/data pipeline with causal preprocessing.
- If `--data` is omitted, it prints `no data — plumbing only` and runs synthetic smoke, without touching STEAD.
- Warm-start defaults to `checkpoints/stage2_distill/best.pt`; smoke run loaded 78/78 tensors, none skipped.
- Optional latency-aware run is separate: `--latency-aware --run-name stage3_causal_latency`.
- Real STEAD fine-tuning was not run in this workspace because dataset paths were not supplied.

## Stage 4 — streaming + true latency plumbing
- Committed `d0287d7`: `causal_stream_probabilities` advances causal preprocessing by chunks, fills unknown future with the latest real processed sample, and emits only newly available samples.
- `scripts/stream_infer.py --causal --checkpoint ... --input raw.npy` runs the causal torch path.
- Verified by `tests/test_streaming.py`: causal chunked output equals fixed-length batch output sample-by-sample.

## Stage 5 — STA/LTA baseline
- `scripts/causal_latency_curve.py` tunes classic STA/LTA over STA/LTA/on-threshold grid on the selected validation/smoke slice.
- STA/LTA is reported as detector-only; phase-classification cells are N/A.

## Stage 6 — headline curve/table [DATASET-GATED]
- Smoke artifacts written in this directory: `latency_curve.csv`, `summary_table.csv`, `recall_latency.png`, `run.json`, `README.md`.
- The smoke run is plumbing-only, not a scientific result: no STEAD training/evaluation and no PNW OOD evaluation were run.
- Real STEAD run: `python scripts/causal_latency_curve.py --data --n <N>` after training `checkpoints/stage3_causal/best.pt`.
- PNW OOD remains held out and should be run through `scripts/pnw_zeroshot.py`; do not train on PNW.
- Smoke ONNX export completed: `stage3_causal_smoke.onnx`, dummy parity max abs error `2.91e-07`.
- Smoke INT8 quantization completed: `stage3_causal_smoke_int8.onnx` via `scripts/quantize_onnx.py --causal --smoke`; random calibration only, no STEAD/PNW evaluation.
- ONNX/INT8 export of a real causal checkpoint remains dataset/checkpoint-gated; rerun `scripts/export_onnx.py --causal` and `scripts/quantize_onnx.py --causal` once `checkpoints/stage3_causal/best.pt` exists.

---

## REAL RUN (STEAD) — supersedes the smoke caveats above

### Preprocessing bug found + fixed (BLOCKER)
- First real causal run collapsed: `checkpoints/stage3_causal/best.pt` emitted a **byte-identical output for every input** in `eval()` mode (cross-trace std = 0.0, P never crossed threshold) -> 0 % streaming recall at every latency budget. Low training P-BCE was the class-imbalance trap (a constant near-zero P scores low BCE because P targets are ~0 on ~99 % of samples).
- Root cause isolated: the model was **healthy in `train()` mode** (batch BN stats: std 0.21, P peaks at onset) and dead only in `eval()` mode. The causal running normalization used the central variance `sqrt(E[x^2]-E[x]^2)`, which collapses to ~0 over the first few samples; dividing by `scale+1e-8` emitted ~1e8 spikes on ~1.7 % of STEAD-train traces, poisoning `stem.1` BatchNorm's running_var to ~3e10 -> eval divides by ~1e5 -> constant. NOT lr, architecture, warm-start, or the streaming path (random-init and warm-start-only causal models were both input-dependent).
- Fix: running **RMS** `sqrt(E[x^2])` in `causal_normalize()` and `CausalPreprocessor._causal_normalize_chunk()` (`src/seismic_edge_picker/preprocessing.py`). == std for the zero-mean post-demean stream in steady state, stays causal + chunk-invariant, cannot collapse. Bounds max |x| 6.5e8 -> 48; median 15.8 -> 15.3 unchanged. All 48 causal/preproc/streaming tests pass. (This is the Stage-2 "warm-start normalization from real background, never zeros" behavior that had only ever been implemented for the bandpass, not the normalizer.)
- Added trainer overrides `--lr` and `--from-scratch` (`scripts/train_distill.py`); the default lr 5e-4 is fine once preprocessing is fixed.

### Primary run (strictly causal, L=0)
- `scripts/train_distill.py --causal --data`, warm-start 78/78 from stage2, 20 epochs, best epoch 19, **val 0.02457** (det 0.112, P 0.0146, S 0.0170) — detection BCE now matches the non-causal stage2 model (0.115). Eval collapse-check: cross-trace std 0.349 (healthy).

### Optional ablation (latency-aware P loss)
- `--latency-aware --run-name stage3_causal_latency`, 20 epochs, best epoch 19, val 0.02456. Latency-aware loss did NOT move latency (median still 3.99 s); it only traded precision (0.986->0.964) for recall (0.972->0.988), tripling false triggers (3.1->9.0/hr). Pure-causal model is preferred.

### Headline curve + table (STEAD-test, n=2000) — `causal_latency_curve.py --data --split test --n 2000`
- Recall vs onset-latency and the metrics table are in the README "Causal early-firing variant" section, `summary_table.csv`, `latency_curve.csv`, `recall_latency.png`.
- Key finding: causality did NOT recover low-latency P-recall (median ~4 s, ~0 recall at 0.5 s). STA/LTA remains the low-latency trigger (0.58 recall @0.5 s, 0.25 s median). Causal U-Net wins on precision/F1/false-alarms/onset-MAE. Confirmed intrinsic by both the pure-causal and latency-aware runs.

### Export (real causal checkpoint)
- `stage3_causal.onnx` parity max-abs-err 1.2e-6 (PASS); INT8 `stage3_causal_int8.onnx` 0.224 -> 0.119 MB (-47 %), FP32 F1 0.992 -> INT8 0.981 (Δ-0.011). Reports: `quantization_report.json`, `int8_eval/`.

### PNW OOD (zero-shot, never trained on PNW)
- Streamed remotely via `scripts/pnw_zeroshot.py` (extended with `--causal`/`--checkpoint`); ~58 MB metadata cached, waveforms streamed on demand. Numbers in README OOD table (`outputs/pnw_zeroshot/` shipped, `outputs/pnw_zeroshot_causal/` causal). Both generalize with a modest F1 drop; the causal model's onset timing/S-pick rate degrade more than the shipped model. Batch eval only (no PNW streaming-latency loader), so the OOD column is detection/pick quality, not latency.
