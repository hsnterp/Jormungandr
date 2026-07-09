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
- ONNX/INT8 export of a real causal checkpoint remains dataset/checkpoint-gated; reuse `scripts/export_onnx.py` and `scripts/quantize_onnx.py` once `checkpoints/stage3_causal/best.pt` exists.
