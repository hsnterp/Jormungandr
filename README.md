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
│   ├── dataset.py              # SeisBench/STEAD torch Dataset (lazy download)
│   └── model.py                # 1D U-Net + param/FLOP counters
├── scripts/
│   ├── inspect_model.py        # Phase 2 verification (param count + MFLOPs)
│   └── sanity_check_data.py    # Phase 1 verification (plot traces + labels)
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
```

## Roadmap / status

| phase | scope | status |
|---|---|---|
| 1 | data pipeline (preprocess, labels, augment, grouped split, sanity plot) | ✅ complete & verified on STEAD |
| 2 | 1D U-Net (<300k params, quant-friendly) | ✅ complete & verified |
| 3 | training (supervised BCE → EQT distillation) | not started |
| 4 | evaluation (F1, pick MAE/std in ms, SNR buckets, EQT comparison) | not started |
| 5 | deployment (ONNX, INT8, latency bench, streaming) | not started |

See [`notes/PROGRESS.md`](notes/PROGRESS.md) for detailed status and the
continuation plan.

## Results

_Populated in Phase 4 — side-by-side vs pretrained EQTransformer (F1, pick MAE,
param count) will be written to `outputs/comparison.md` and inlined here._
