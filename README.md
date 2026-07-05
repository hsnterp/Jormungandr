# Jormungandr

A compact, **edge-deployable seismic event detector and phase picker**. It
takes a 60-second, 3-channel, 100 Hz waveform and outputs three per-sample
probability streams — **event detection**, **P-arrival**, **S-arrival** — from a
1D U-Net small enough to run **real-time INT8 inference on a Raspberry Pi**.

The model is trained on [STEAD](https://github.com/smousavi05/STEAD) via
[SeisBench](https://github.com/seisbench/seisbench) and distilled from a
pretrained **EQTransformer** teacher. The repository and distribution are named
**Jormungandr**; the stable Python import package remains `seismic_edge_picker`.

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

## Live demo

**▶ [Interactive streaming demo](https://hsnterp.github.io/Jormungandr/docs/demo.html)**
— replays six STEAD test traces as a live left-to-right stream: the raw
3-channel waveform arrives while the student model's detection / P / S
probability streams respond in sync, with ground-truth arrivals (dashed) and
EQTransformer teacher picks (dotted) overlaid and a per-trace latency /
confidence readout. Cases include a clean high-SNR event, a near-threshold
low-SNR event, pure noise (correct true-negative), and a failure case (8.9 s P
miss). Self-contained page in [`docs/demo.html`](docs/demo.html), data in
[`outputs/figures/demo_traces.json`](outputs/figures/demo_traces.json)
(exported by `scripts/export_demo_traces.py`).

> GitHub Pages is served from `main` (root, via Jekyll), so the demo publishes at
> the `/docs/demo.html` path above a minute or two after this commit lands. You
> can also open `docs/demo.html` locally (it embeds an offline copy of the data),
> or serve the folder with `python -m http.server`.

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
Jormungandr/
├── configs/default.yaml        # single source of truth for all phases
├── src/seismic_edge_picker/
│   ├── config.py               # YAML → attribute namespace
│   ├── preprocessing.py        # demean, bandpass, normalize
│   ├── labels.py               # arrival samples → (3,6000) target masks
│   ├── augment.py              # window-shift, noise-mix, channel-dropout (train)
│   ├── splits.py               # grouped, leakage-free train/val/test split
│   ├── dataset.py              # cached-only SeisBench/STEAD torch Dataset
│   ├── losses.py               # weighted per-stream BCE
│   ├── model.py                # 1D U-Net + param/FLOP counters
│   └── streaming.py            # overlap merge + event/P/S postprocessing
├── scripts/
│   ├── inspect_model.py        # Phase 2 verification (param count + MFLOPs)
│   ├── sanity_check_data.py    # Phase 1 verification (plot traces + labels)
│   ├── train.py                # Stage-1 training + tiny smoke mode
│   ├── evaluate.py             # Phase 4: F1 + pick residuals on a split
│   ├── threshold_sweep.py      # Phase 4: detection threshold + min-duration sweep
│   ├── eqtransformer_baseline.py  # Phase 4: pretrained EQTransformer side-by-side
│   ├── cache_teacher.py        # Stage 2a: chunked/resumable teacher-output cache
│   ├── train_distill.py        # Stage 2b: distillation fine-tune (hard+soft blend)
│   ├── distill_smoke.py        # Stage 2: tiny end-to-end cache + 1-epoch smoke
│   ├── export_onnx.py          # Phase 5: ONNX export + ORT parity check
│   ├── quantize_onnx.py        # Phase 5b: INT8 static quantization + parity/eval
│   ├── benchmark_latency.py    # Phase 5c: CPU latency + throughput report
│   └── stream_infer.py         # Phase 5d: continuous INT8 ONNX inference demo
├── src/seismic_edge_picker/distill.py  # Stage 2 loss + cache + teacher loading
├── docs/stage2.md              # Stage 2 pipeline, loss, cache format, commands
├── tests/                      # pytest sanity tests (run without the dataset)
├── docs/PROGRESS.md            # compacted status, metrics, commands, next steps
└── configs / data / checkpoints / outputs
```

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```

`requirements.txt` pins the exact Python 3.12 environment used for the published
verification. Training uses a GPU; **all inference/deployment code is CPU-only**.

## Published model artifacts

The repository ships the directly usable FP32 and INT8 ONNX deployment models:

- `outputs/onnx/stage2_distill.onnx`
- `outputs/onnx/stage2_distill_int8.onnx` (default deployment artifact)

STEAD data and PyTorch checkpoints are intentionally excluded. Commands that
reference `checkpoints/stage2_distill/best.pt` require reproducing Stage 1/Stage 2
training first. ONNX benchmarking and streaming work directly from the included
artifacts without a checkpoint. See [`MODEL_CARD.md`](MODEL_CARD.md) for intended
use, evaluation, and limitations.

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

# Full Stage 1 (50 epochs configured; expensive GPU run)
python scripts/train.py --config configs/default.yaml

# Phase 4 — evaluate a checkpoint on the test split
python scripts/evaluate.py --config configs/default.yaml \
    --checkpoint checkpoints/stage1/best.pt --out outputs/stage1_eval

# Stage 2 — tiny distillation smoke (cache ~24 teacher outputs + 1 epoch; cheap)
python scripts/distill_smoke.py --config configs/default.yaml

# Stage 2 — full distillation (EXPENSIVE, GPU) — already run, see docs/stage2.md
python scripts/cache_teacher.py --config configs/default.yaml            # 2a: cache EQT teacher outputs (chunked/resumable)
python scripts/train_distill.py --config configs/default.yaml \
    --init checkpoints/stage1/best.pt                                    # 2b: distillation fine-tune (warm-start from Stage 1)
python scripts/evaluate.py --config configs/default.yaml \
    --checkpoint checkpoints/stage2_distill/best.pt --out outputs/stage2_eval
python scripts/threshold_sweep.py --config configs/default.yaml \
    --checkpoint checkpoints/stage2_distill/best.pt --out outputs/stage2_eval
```

## Roadmap / status

| phase | scope | status |
|---|---|---|
| 1 | data pipeline (preprocess, labels, augment, grouped split, sanity plot) | ✅ complete & verified on STEAD |
| 2 | 1D U-Net (<300k params, quant-friendly) | ✅ complete & verified |
| 3 | training (supervised BCE → EQT distillation) | ✅ Stage 1 (50 epochs) **and** Stage 2 distillation complete |
| 4 | evaluation (F1, pick MAE/std in ms, SNR buckets, EQT comparison) | ✅ distilled student evaluated + threshold-tuned; EQT side-by-side done |
| 5 | deployment (ONNX, INT8, latency bench, streaming) | ✅ **complete** |

See [`docs/PROGRESS.md`](docs/PROGRESS.md) for detailed status and the
continuation plan.

## Results

**Headline model: the Stage 2 EQTransformer-distilled student.** The training
checkpoint is reproduced at `checkpoints/stage2_distill/best.pt`; the included
public deployment artifact is `outputs/onnx/stage2_distill_int8.onnx`. Everything
below is evaluated on the
**identical** held-out test split (7,781 traces: 4,957 earthquake / 2,824 noise)
with the **identical** detection/pick tolerances (peak height 0.3, match
tolerance ±500 ms), so every row is directly comparable.

### Headline comparison (test split)

| model / operating point | params | precision | recall | **F1** | FP (noise) | FN (eq) | P MAE±std | S MAE±std |
|---|---|---|---|---|---|---|---|---|
| **Stage 2 distilled — max-F1 (thr 0.80)** | **48,051** | 0.9910 | 0.9978 | **0.9944** | 45 | 11 | **36.9±56.3 ms** | **71.4±109.4 ms** |
| **Stage 2 distilled — low-false-alarm (thr 0.90 + 500 ms)** | 48,051 | 0.9977 | 0.9831 | 0.9903 | **11** | 84 | 36.9±56.3 ms | 71.4±109.4 ms |
| Stage 2 distilled — default (thr 0.50) | 48,051 | 0.9649 | 0.9994 | 0.9819 | 180 | 3 | 36.9±56.3 ms | 71.4±109.4 ms |
| Stage 1 supervised — max-F1 (thr 0.90 + 500 ms) | 48,051 | 0.9894 | 0.9964 | 0.9929 | 53 | 18 | 46.0±77.4 ms | 72.4±112.3 ms |
| Stage 1 supervised — default (thr 0.50) | 48,051 | 0.9227 | 0.9998 | 0.9597 | 415 | 1 | 46.0±77.4 ms | 72.4±112.3 ms |
| EQTransformer teacher † (thr 0.50) | 376,935 | 0.993 | 0.979 | 0.9860 | 35 | 103 | 62.8±88.4 ms | 78.9±117.5 ms |
| EQTransformer teacher † (best-F1 @ 0.15) | 376,935 | 0.984 | 0.990 | 0.9867 | 81 | 51 | 62.8±88.4 ms | 78.9±117.5 ms |

† These two EQTransformer rows are the **handicapped baseline** — teacher fed the
student's preprocessing at a fixed/test-picked threshold. Its **fair** numbers
(native preprocessing, validation-selected threshold: F1 0.9994, P/S-MAE
41.1/73.6 ms) are in **Fairness corrections** below.

**What distillation bought (same 48,051-param model, only better weights):** at
the config-default threshold 0.50, distillation more than halved noise false
alarms (**415 → 180**), lifting F1 **0.9597 → 0.9819**, and tightened P picks
(MAE **46.0 → 36.9 ms**, S 72.4 → 71.4 ms) — the precision gain the teacher was
expected to transfer. After the free threshold + min-duration postprocessing, the
distilled student reaches **F1 0.9944** (45 noise FP, 11 missed events), beating
both the Stage 1 tuned student (0.9929) and the **7.8×-larger** EQTransformer
teacher's *handicapped* best (0.9867) — but that teacher number was measured
under the student's preprocessing and an un-tuned threshold; see **Fairness
corrections** below, where the teacher reaches F1 0.9994 under its native input.
Its low-false-alarm operating point drives noise
false alarms down to **11 / 2,824** (0.4%) while still recovering 98.3% of events.
Both models' val weighted BCE is ~0.0179 — the headline gains are in the shape of
the detection/pick streams, not the aggregate loss. Full metrics, SNR-bucket
breakdown, and residual histograms: `outputs/stage2_eval/` (`test_metrics.json`,
`snr_breakdown.csv`, `pick_residuals.png`, `summary.txt`,
`threshold_recommendations.json`); Stage 1's equivalents remain in
`outputs/stage1_eval/`.

### Fairness corrections (validation-selected threshold + native teacher preprocessing)

Two protocol fixes make the student-vs-teacher comparison fair. Both are applied
in `scripts/fair_comparison.py`; the corrected numbers live in
`outputs/fair_eval/comparison.json` and drive the regenerated
`outputs/figures/snr_bucketed_performance.png`.

1. **Threshold selected on validation, not test.** The tables above and the
   original SNR chart scored detection at a fixed, a-priori threshold (0.50) —
   and the "max-F1" rows picked the threshold on the *test* split itself. The
   corrected protocol sweeps the detection threshold on the **validation** split
   only, fixes each model at its val-max-F1 point (**student 0.89, teacher
   0.89**), and evaluates the **test** split once at that fixed threshold.
2. **Each model under its own preprocessing.** The original teacher was fed the
   *student's* pipeline (demean + bandpass 1–45 Hz + global-std norm).
   EQTransformer's SeisBench `stead` weights document their own conditioning
   (demean + per-trace **peak** normalization + 6-sample cosine taper, **no
   bandpass**); the teacher is now fed that native input via the model's own
   `annotate_batch_pre`. The student's evaluation is unchanged.

Effect on the overall test-split numbers (detection F1 incl. noise; P/S-pick MAE
within ±500 ms):

| protocol | student F1 | student P-MAE | student S-MAE | teacher F1 | teacher P-MAE | teacher S-MAE |
|---|---|---|---|---|---|---|
| published (fixed thr 0.50; teacher on student pipeline) | 0.9819 | 36.9 ms | 71.4 ms | 0.9860 | 62.8 ms | 78.9 ms |
| + fix 1 (val-selected thr; teacher still on student pipeline) | **0.9925** | 36.9 ms | 71.4 ms | 0.9863 | 62.8 ms | 78.9 ms |
| + fix 1 & 2 (val-selected thr; **teacher native** preprocessing) | 0.9925 | **36.9 ms** | **71.4 ms** | **0.9994** | **41.1 ms** | **73.6 ms** |

Fix 1 mainly helps the **student** — its confident detection wants a high
threshold to shed noise false alarms (F1 0.9819 → 0.9925). Fix 2 mainly helps
the **teacher** — native input lifts it from F1 0.9863 → **0.9994**, P-MAE
62.8 → **41.1 ms**, and S-MAE 78.9 → **73.6 ms**.

**Honest revised takeaway.** Under this fair (and teacher-favorable) protocol the
EQTransformer teacher slightly **leads detection F1** (0.9994 vs 0.9925) and
closes most of the pick-timing gap, yet the **48,051-param student stays ahead on
both pick MAEs** (P 36.9 vs 41.1 ms, S 71.4 vs 73.6 ms) at **7.8× fewer
parameters** and with only INT8-friendly ops. In the regenerated SNR chart the
teacher leads F1 in every earthquake bucket; because those buckets are
earthquake-only, per-bucket F1 tracks recall and the student's noise-false-alarm
advantage shows up only in the overall figure here.

### Size / cost comparison

| model | params | fp32 size | MFLOPs / 60 s window | throughput (A100, fp32) | ops |
|---|---|---|---|---|---|
| **SeismicUNet student** (Stage 1 & 2) | **48,051** | **0.19 MB** | **38.1** | ~640 tr/s | Conv1d / BN / ReLU / NN-upsample (INT8-friendly) |
| EQTransformer teacher | 376,935 | 1.51 MB | — | ~540 tr/s | attention + BiLSTM (not edge-friendly) |

The student is **7.8× smaller** and faster than the teacher, stays competitive
with it on detection and ahead on both P- and S-pick timing under a fair,
teacher-favorable protocol (see **Fairness corrections**), and — unlike the teacher — uses only
quantization-friendly ops, so it is the model carried into Phase 5 (ONNX / INT8)
deployment.

### Detection threshold + min-duration tuning (postprocessing only)

`scripts/threshold_sweep.py` sweeps the detection threshold (0.10→0.90) and a
lightweight **minimum-duration** postprocessing rule (a trace counts as detected
only if the detection stream stays above threshold for ≥ N consecutive samples).
This is pure postprocessing — no retrain, no model change. Operating points below
are for the **headline Stage 2 distilled** checkpoint on the test split:

| operating point | threshold | min-duration | precision | recall | **F1** | FP (noise) | FN (eq) |
|---|---|---|---|---|---|---|---|
| config default (`eval.detection_threshold`) | 0.50 | none | 0.9649 | 0.9994 | 0.9819 | 180 | 3 |
| **max-F1** | **0.80** | none | 0.9910 | 0.9978 | **0.9944** | 45 | 11 |
| **low-false-alarm** | **0.90** | **500 ms** | 0.9977 | 0.9831 | 0.9903 | **11** | 84 |
| recall-preserving | 0.25 | 100 ms | 0.9222 | 0.9998 | 0.9594 | 418 | **1** |

Distillation already cleaned up most noise false alarms, so — unlike the Stage 1
student, which needed thr 0.90 + a 500 ms rule to reach its best F1 — the distilled
student peaks at just **thr 0.80 with no min-duration rule** (F1 0.9944, 45 FP).
The min-duration rule now mainly serves the **low-false-alarm** point: thr 0.90 +
500 ms drives noise false alarms to **11 / 2,824** (0.4%) while still recovering
98.3% of events. If zero missed events dominate, the recall-preserving point holds
recall at 0.9998 (1 miss) at the cost of more false alarms. Sweep artifacts:
`outputs/stage2_eval/threshold_sweep.csv`, `threshold_sweep.png`,
`threshold_recommendations.json` (Stage 1's equivalents remain in
`outputs/stage1_eval/`).

Run it with:

```bash
python scripts/threshold_sweep.py --config configs/default.yaml \
    --checkpoint checkpoints/stage2_distill/best.pt --out outputs/stage2_eval
```

### Side-by-side vs pretrained EQTransformer (teacher)

This is the **pre-distillation, student-pipeline** baseline that motivated Stage 2.
Its teacher numbers are handicapped (EQT fed the student's preprocessing); the
**fair** re-scoring is in **Fairness corrections** above. `scripts/eqtransformer_baseline.py`
runs SeisBench's pretrained EQTransformer (`stead` weights) on the **identical**
test traces with the **identical** detection/pick tolerances (`eval.*`), so the
numbers line up. Both models are fed byte-identical inputs — the project pipeline's
demean + bandpass(1–45 Hz) + std normalization. Student figures here are the
Stage 1 (supervised, pre-distillation) checkpoint at detection threshold **0.50**:

| model | params | fp32 size | P | R | **F1** | FP (noise) | FN (eq) | P MAE±std | S MAE±std | throughput |
|---|---|---|---|---|---|---|---|---|---|---|
| **SeismicUNet** (student, default 0.50) | **48,051** | **0.19 MB** | 0.923 | 1.000 | 0.9597 | 415 | 1 | **46.0±77.4 ms** | **72.4±112.3 ms** | ~660 tr/s |
| SeismicUNet (student, tuned 0.90+500 ms) | 48,051 | 0.19 MB | 0.989 | 0.996 | **0.9929** | 53 | 18 | 46.0±77.4 ms | 72.4±112.3 ms | ~660 tr/s |
| EQTransformer † (`stead`, 0.50) | 376,935 | 1.51 MB | **0.993** | 0.979 | 0.9860 | **35** | 103 | 62.8±88.4 ms | 78.9±117.5 ms | ~540 tr/s |
| EQTransformer † (`stead`, best-F1 @ 0.15) | 376,935 | 1.51 MB | 0.984 | 0.990 | 0.9867 | 81 | 51 | 62.8±88.4 ms | 78.9±117.5 ms | ~540 tr/s |

Takeaways (on this **student-pipeline** baseline / test set): the **7.8×-smaller
student is competitive with the teacher**. At a matched threshold EQTransformer
has far higher raw precision (35 vs 415 noise false alarms) but lower recall
(misses 103 events vs 1). **† The EQT rows here are handicapped** — fed the
student's preprocessing at a fixed/test-picked threshold. Under its **native**
preprocessing and a **validation-selected** threshold the teacher reaches F1
**0.9994** and P/S-MAE **41.1 / 73.6 ms** (see **Fairness corrections**),
**edging the student on detection F1** while the 48k-param student keeps the
pick-MAE (P 36.9, S 71.4 ms) and 7.8× size lead. Artifacts:
`outputs/eqtransformer_baseline/` (`eqt_metrics.json`, `threshold_sweep.csv/png`,
`pick_residuals.png`, `summary.txt`).

**Fairness caveats — do not over-read these numbers:**
- **Preprocessing (now addressed):** the rows in *this* table feed EQT the
  project's bandpass(1–45 Hz)+std-normalized inputs, **not** its native
  preprocessing — which handicaps its pick sharpness. The **Fairness corrections**
  section above re-scores the teacher under its native pipeline (demean +
  per-trace peak-norm + taper, no bandpass), removing this caveat.
- **Windowing:** single fixed 60 s windows, no overlap/stacking. EQT's usual
  `classify()` uses overlapping windows + stacking on continuous streams; both
  models here run one window at a time (the student's deployment setting).
- **Thresholds:** EQT's native default detection threshold is 0.1 (swept here;
  best F1 at 0.15). The 0.50 row matches the student's default for alignment.
- Labels (STEAD arrival samples), 3-component ZNE order, 100 Hz, and 6000-sample
  windows are identical for both, so those axes are apples-to-apples.

Run it with:

```bash
python scripts/eqtransformer_baseline.py --config configs/default.yaml \
    --out outputs/eqtransformer_baseline
```

## Phase 5 deployment (ONNX / INT8)

The distilled student is the model that ships. All Phase 5 code is CPU-only;
`onnx` (1.22.0) and `onnxruntime` (1.27.0) are installed.

### ✅ ONNX export + parity (done)

`scripts/export_onnx.py` reuses `build_model` + the config + the same
`weights_only` safe-load as `evaluate.py` (no model redefinition), exports the
Stage 2 distilled student, and checks ONNX Runtime vs PyTorch parity:

```bash
python scripts/export_onnx.py --config configs/default.yaml \
    --checkpoint checkpoints/stage2_distill/best.pt \
    --out outputs/onnx/stage2_distill.onnx
```

- **Export:** opset **17**, legacy TorchScript exporter (`dynamo=False`, no extra
  deps), dynamic batch axis, named I/O (`waveform` → `streams`). Artifact
  `outputs/onnx/stage2_distill.onnx` (0.20 MB); `onnx.checker` passes.
- **Parity (PyTorch eval, `torch.no_grad` vs ONNX Runtime CPU):**

  | input | output shape | max abs err | mean abs err |
  |---|---|---|---|
  | dummy `(1,3,6000)` | `(1,3,6000)` | 1.11e-07 | 2.41e-08 |
  | real test batch `(8,3,6000)` | `(8,3,6000)` | 7.45e-07 | 3.31e-08 |

  Both are far under the 1e-4 tolerance; the batch-8 run confirms the dynamic
  batch axis. Report saved to `outputs/onnx/stage2_distill_parity.json`.

### ✅ INT8 static quantization + evaluation (done)

`scripts/quantize_onnx.py` calibrates ONNX Runtime static QDQ quantization on
500 validation traces, compares PyTorch / FP32 ONNX / INT8 ONNX outputs, and
evaluates FP32 and INT8 ONNX on the test split:

```bash
python scripts/quantize_onnx.py --config configs/default.yaml \
    --checkpoint checkpoints/stage2_distill/best.pt \
    --fp32-onnx outputs/onnx/stage2_distill.onnx \
    --int8-out outputs/onnx/stage2_distill_int8.onnx \
    --threshold 0.80
```

- **Quantization:** QDQ, per-channel QInt8 weights, QUInt8 activations, MinMax
  calibration. Both full eligible-op INT8 and body-INT8/head-FP32 variants were
  tested. Detection metrics were identical; full INT8 had marginally lower
  combined pick MAE and was shipped.
- **Size:** 0.204 MB FP32 → 0.104 MB INT8: **1.95× smaller, 48.8% reduction**.
- **FP32 ONNX vs INT8 parity:**

  | input | output shape | max abs err | mean abs err |
  |---|---|---|---|
  | dummy `(1,3,6000)` | `(1,3,6000)` | 1.47e-02 | 9.30e-04 |
  | real test batch `(8,3,6000)` | `(8,3,6000)` | 9.24e-01 | 2.59e-02 |

- **Full test split at the Stage 2 threshold (0.80):**

  | model | detection F1 | precision | recall | FP | FN | P MAE | S MAE |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | FP32 ONNX | 0.9944 | 0.9910 | 0.9978 | 45 | 11 | 37.0 ms | 71.4 ms |
  | INT8 ONNX | 0.9920 | 0.9941 | 0.9899 | 29 | 50 | 45.6 ms | 76.2 ms |

Quantization does not meaningfully hurt this operating point: F1 falls 0.0024,
P-pick MAE rises 8.7 ms, and S-pick MAE rises 4.8 ms. Reports are saved to
`outputs/onnx/quantization_report.json` and
`outputs/stage2_int8_eval/int8_eval.json`.

### ✅ CPU latency benchmark (done)

```bash
python scripts/benchmark_latency.py --config configs/default.yaml \
    --checkpoint checkpoints/stage2_distill/best.pt \
    --fp32-onnx outputs/onnx/stage2_distill.onnx \
    --int8-onnx outputs/onnx/stage2_distill_int8.onnx \
    --out-dir outputs/latency
```

Measured on an AMD EPYC 7763 with input `(1,3,6000)`, one CPU thread, ORT
sequential execution, 20 excluded warmups, and 200 measured runs per backend:

| backend | p50 | p95 | mean | throughput |
|---|---:|---:|---:|---:|
| PyTorch CPU | 3.149 ms | 4.119 ms | 3.371 ms | 296.7 windows/s |
| FP32 ONNX Runtime | 1.543 ms | 1.571 ms | 1.567 ms | 638.0 windows/s |
| INT8 ONNX Runtime | **1.226 ms** | **1.253 ms** | **1.233 ms** | **811.0 windows/s** |

INT8 ONNX is 1.27× faster than FP32 ONNX and 2.73× faster than PyTorch by
mean latency on this host. Results are hardware-specific;

Raspberry Pi CPU result, measured on-device with one ONNX Runtime CPU thread:

| backend | p50 | p95 | mean | throughput |
|---|---:|---:|---:|---:|
| FP32 ONNX Runtime | 3.095 ms | 3.164 ms | 3.106 ms | 322.0 windows/s |
| INT8 ONNX Runtime | 2.392 ms | 2.400 ms | 2.392 ms | 418.1 windows/s |

On Raspberry Pi, the INT8 model processes one 60 s, 3-channel window in 2.392 ms using one CPU thread. With the default 30 s streaming hop, this is over 12,000× faster than real time.

rerun the same command
on Raspberry Pi or Graviton targets. Reports:
`outputs/latency/latency_report.json` and `outputs/latency/latency_report.md`.
CUDA was not benchmarked because Phase 5 targets CPU deployment.

### ✅ Streaming inference wrapper (done)

`src/seismic_edge_picker/streaming.py` provides fixed-hop window generation,
zero-padded tail handling, uniform overlap averaging, contiguous event extraction,
short-gap coalescing, P/S peak extraction, and event association. The CLI uses
the INT8 ONNX model and one ORT CPU thread by default:

```bash
python scripts/stream_infer.py --demo-traces 4 --plot --save-probabilities \
    --out-dir outputs/streaming_demo
```
This was also smoke-tested on a Raspberry Pi with a 240 s synthetic 3-channel signal. The INT8 ONNX streaming path ran end-to-end on CPU and produced zero events on random-noise input, as expected.

The smoke demo concatenated four test traces (two earthquake, two noise) into a
240 s signal. Seven 60 s windows at a 30 s hop produced **2 coalesced events,
2 P picks, and 1 S pick**. This verifies the streaming path; it is not a new
accuracy evaluation. Outputs:

- `events.csv` and `picks.csv` — relative timestamps and probabilities;
- `summary.json` / `summary.txt` — settings, source traces, counts, and records;
- `merged_probabilities.npz` — optional merged streams and overlap coverage;
- `streaming_predictions.png` — optional waveform/probability visualization.

For a continuous raw float32 array shaped `(3,N)` or `(N,3)` at 100 Hz:

```bash
python scripts/stream_infer.py --input continuous.npy \
    --out-dir outputs/streaming_demo
```

Raw arrays are demeaned, bandpassed, and normalized per 60 s model window;
`--input-preprocessed` skips that step. Output times are seconds relative to the
array start. The default detection point is threshold **0.80**, minimum duration
**10 ms**, with qualifying fragments separated by at most 0.5 s coalesced into
one event. P/S peaks default to 0.30 and are event-gated; use
`--emit-unassociated-picks` to retain all candidates.

The FP32 **0.90 + 500 ms** low-false-alarm point has not been separately retuned
for merged INT8 streaming output, so it is documented as an override—not claimed
as a validated INT8 streaming operating point. The model uses only INT8-friendly
ops (`Conv1d` / `BatchNorm1d` / `ReLU` / NN-upsample).


## Citation and licensing

Jormungandr is released under the [GNU GPL v3](LICENSE). The shipped model was
trained on STEAD and distilled from EQTransformer through SeisBench. If you use
this project, cite Jormungandr via [`CITATION.cff`](CITATION.cff) and cite the
underlying work:

- Mousavi et al. (2019), **STEAD**, DOI
  [`10.1109/ACCESS.2019.2947848`](https://doi.org/10.1109/ACCESS.2019.2947848)
  — dataset licensed CC BY 4.0.
- Mousavi et al. (2020), **EQTransformer**, DOI
  [`10.1038/s41467-020-17591-w`](https://doi.org/10.1038/s41467-020-17591-w).
- Woollam et al. (2022), **SeisBench**, DOI
  [`10.1785/0220210324`](https://doi.org/10.1785/0220210324).

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for complete attribution,
license links, and modification notices.
