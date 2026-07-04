# seismic-edge-picker

A compact, **edge-deployable seismic event detector and phase picker**. It
takes a 60-second, 3-channel, 100 Hz waveform and outputs three per-sample
probability streams — **event detection**, **P-arrival**, **S-arrival** — from a
1D U-Net small enough to run **real-time INT8 inference on a Raspberry Pi**.

The model is trained on [STEAD](https://github.com/smousavi05/STEAD) via
[SeisBench](https://github.com/seisbench/seisbench) and distilled from a
pretrained **EQTransformer** teacher.

## Motivation

Seismic phase picking is a real-time signal-processing problem: a continuous
multichannel stream in, precisely-timed event/onset labels out. The same shape
of problem shows up in **robotics and embedded sensing** — contact/event
detection, onset timing, and streaming segmentation on hardware with a tight
compute and power budget. This project is a focused study of taking a heavy
research model (EQTransformer) and compressing it into something that:

- fits in **<300k parameters** with only **quantization-friendly ops**,
- runs **CPU-only** at the edge with predictable latency,
- and holds up against the teacher on detection F1 and pick accuracy.

## Task formulation

| | |
|---|---|
| **Input** | `(3, 6000)` — 3 components, 60 s @ 100 Hz |
| **Output** | `(3, 6000)` sigmoid streams: detection / P / S |
| detection | 1.0 across the event window (P → coda) |
| P / S | Gaussian bump (σ ≈ 0.25 s) centered on the arrival |
| noise | all-zero target |

## Architecture

```
input  (3, 6000)
  │  stem: conv k7 s2  + BN + ReLU
  ▼
 s0 (16, 3000) ───────────────────────────────────skip──────────────┐
  │  enc1: [dw k7 s2 → pw → BN → ReLU]                               │
  ▼                                                                  │
 e1 (16, 1500) ──────────────────────────────skip────────┐          │
  │  enc2                                                 │          │
  ▼                                                       │          │
 e2 (32, 750) ─────────────────────────skip───┐          │          │
  │  enc3                                      │          │          │
  ▼                                            │          │          │
 e3 (64, 375) ──────────────────skip┐         │          │          │
  │  enc4                           │         │          │          │
  ▼                                 │         │          │          │
 e4 (96, 188)                       │         │          │          │
  │  bottleneck: 2× dw-sep, dil 2,4 │         │          │          │
  ▼                                 │         │          │          │
 b  (96, 188)                       │         │          │          │
  │  dec1: NN-up→375, cat e3, ds ◄──┘         │          │          │
  ▼                                           │          │          │
    (64, 375)                                 │          │          │
  │  dec2: NN-up→750, cat e2, ds ◄────────────┘          │          │
  ▼                                                      │          │
    (32, 750)                                            │          │
  │  dec3: NN-up→1500, cat e1, ds ◄─────────────────────-┘          │
  ▼                                                                 │
    (16, 1500)                                                      │
  │  dec4: NN-up→3000, cat s0, ds ◄────────────────────────────────-┘
  ▼
    (16, 3000)
  │  NN-up→6000, head 1x1 conv, sigmoid
  ▼
output (3, 6000)
```

- **Only INT8-friendly ops**: `Conv1d`, `BatchNorm1d`, `ReLU`, nearest-neighbor
  `interpolate`. No LSTM / attention / transposed conv.
- Depthwise-separable blocks throughout keep the parameter count tiny.

**Measured at init** (`scripts/inspect_model.py`):

| metric | value |
|---|---|
| parameters | **48,051** (budget < 300k ✅) |
| MFLOPs / 60 s window | **38.1** |
| output range | `[0,1]` sigmoid ✅ |

## Repository layout

```
seismic-edge-picker/
├── configs/default.yaml        # single source of truth for all phases
├── src/seismic_edge_picker/
│   ├── config.py               # YAML → attribute namespace
│   ├── preprocessing.py        # demean, bandpass, normalize
│   ├── labels.py               # arrival samples → (3,6000) target masks
│   ├── augment.py              # window-shift, noise-mix, channel-dropout (train)
│   ├── splits.py               # grouped, leakage-free train/val/test split
│   ├── dataset.py              # cached-only SeisBench/STEAD torch Dataset
│   ├── losses.py               # weighted per-stream BCE
│   └── model.py                # 1D U-Net + param/FLOP counters
├── scripts/
│   ├── inspect_model.py        # Phase 2 verification (param count + MFLOPs)
│   ├── sanity_check_data.py    # Phase 1 verification (plot traces + labels)
│   ├── train.py                # Stage-1 training + tiny smoke mode
│   ├── evaluate.py             # Phase 4: F1 + pick residuals on a split
│   └── threshold_sweep.py      # Phase 4: detection threshold + min-duration sweep
├── tests/                      # pytest sanity tests (run without the dataset)
├── notes/PROGRESS.md           # phase-by-phase status + handoff notes
└── configs / data / checkpoints / outputs
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Training uses a GPU; **all inference/deployment code is CPU-only**.

## Reproduction

```bash
# Phase 2 — inspect the model (no data needed)
python scripts/inspect_model.py --config configs/default.yaml

# tests (no dataset needed — pure logic)
pytest -q

# Phase 1 — visualize traces + label masks (needs STEAD cached locally)
python scripts/sanity_check_data.py --config configs/default.yaml \
    --n 5 --split train --out outputs/sanity_labels.png

# Stage 1 smoke test (16 train + 8 val traces, one epoch)
python scripts/train.py --config configs/default.yaml --smoke-test

# Full Stage 1 (50 epochs configured; requires explicit approval before launch)
python scripts/train.py --config configs/default.yaml

# Phase 4 — evaluate a checkpoint on the test split
python scripts/evaluate.py --config configs/default.yaml \
    --checkpoint checkpoints/stage1/best.pt --out outputs/stage1_eval
```

## Roadmap / status

| phase | scope | status |
|---|---|---|
| 1 | data pipeline (preprocess, labels, augment, grouped split, sanity plot) | ✅ complete & verified on STEAD |
| 2 | 1D U-Net (<300k params, quant-friendly) | ✅ complete & verified |
| 3 | training (supervised BCE → EQT distillation) | Stage 1 complete (50 epochs); Stage 2 distillation not started |
| 4 | evaluation (F1, pick MAE/std in ms, SNR buckets, EQT comparison) | student evaluated on test split; EQT side-by-side pending |
| 5 | deployment (ONNX, INT8, latency bench, streaming) | not started |

See [`notes/PROGRESS.md`](notes/PROGRESS.md) for detailed status and the
continuation plan.

## Results

### Stage 1 (supervised, 50 epochs) — test split

Best checkpoint `checkpoints/stage1/best.pt` (epoch 48, val weighted BCE
**0.01794**), evaluated on the held-out test split (7,781 traces: 4,957
earthquake / 2,824 noise) with `scripts/evaluate.py` (detection threshold 0.5,
pick-peak height 0.3, match tolerance ±500 ms):

| metric | value |
|---|---|
| test weighted BCE | **0.01770** |
| detection precision / recall / **F1** | 0.9227 / 0.9998 / **0.9597** |
| P pick MAE / std (within ±500 ms; hit rate 97.6%) | **46.0 ms** / 77.4 ms |
| S pick MAE / std (within ±500 ms; hit rate 96.4%) | **72.4 ms** / 112.3 ms |
| parameters | 48,051 |

Precision is limited by false alarms on noise traces (415 / 2,824 = 14.7%
noise false-alarm rate); recall is near-perfect (1 missed event). Pick MAE
degrades gracefully with SNR (P: 42 ms at >20 dB → 66 ms at 0–10 dB). Full
metrics, SNR-bucket breakdown, and residual histograms live in
`outputs/stage1_eval/` (`test_metrics.json`, `snr_breakdown.csv`,
`pick_residuals.png`, `summary.txt`).

### Detection threshold + min-duration tuning (postprocessing only)

`scripts/threshold_sweep.py` sweeps the detection threshold (0.10→0.90) and a
lightweight **minimum-duration** postprocessing rule (a trace counts as detected
only if the detection stream stays above threshold for ≥ N consecutive samples).
This is pure postprocessing — no retrain, no model change. On the test split, F1
rises monotonically with threshold and the min-duration rule further trims noise
false alarms; recommended operating points:

| operating point | threshold | min-duration | precision | recall | **F1** | FP (noise) | FN (eq) |
|---|---|---|---|---|---|---|---|
| config default (`eval.detection_threshold`) | 0.50 | none | 0.923 | 1.000 | 0.9597 | 415 | 1 |
| **max-F1 / lowest false-alarm** | **0.90** | **500 ms** | 0.9894 | 0.9964 | **0.9929** | **53** | 18 |
| recall-preserving alternative | 0.90 | 100 ms | 0.9851 | 0.9978 | 0.9914 | 75 | **11** |

Tuning cuts noise false alarms ~8× (415 → 53) and lifts F1 from 0.960 → 0.993
without touching the model. The max-F1 and minimum-false-alarm points coincide at
threshold 0.90 + 500 ms; the 100 ms variant trades 22 more false alarms for 7
fewer missed earthquakes. If zero missed events matter more than false alarms,
recall stays 1.000 for thresholds ≤ 0.70 (at higher FP). Sweep artifacts:
`outputs/stage1_eval/threshold_sweep.csv`, `threshold_sweep.png`,
`threshold_recommendations.json`.

Run it with:

```bash
python scripts/threshold_sweep.py --config configs/default.yaml \
    --checkpoint checkpoints/stage1/best.pt --out outputs/stage1_eval
```

_Side-by-side vs pretrained EQTransformer (F1, pick MAE, param count) comes
with Phase 4 completion and will be inlined here._
