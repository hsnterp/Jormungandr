# Jormungandr model card

## Model summary

Jormungandr is a compact 48,051-parameter 1D U-Net for three-component seismic
event detection and P/S phase picking. The public deployment artifact is the
Stage 2 distilled INT8 ONNX model at
`outputs/onnx/stage2_distill_int8.onnx`.

- Input: float32 `(batch, 3, 6000)`, ZNE order, 100 Hz, 60 seconds.
- Output: `(batch, 3, 6000)` sigmoid streams for detection, P, and S.
- Architecture: convolutional U-Net with depthwise-separable convolutions,
  nearest-neighbor upsampling, batch normalization, and ReLU.
- Teacher: pretrained EQTransformer `stead` model loaded through SeisBench.
- Training data: a grouped 75,000-trace STEAD subset (50,000 earthquake and
  25,000 noise traces), split by station/event to reduce leakage.

## Shipped artifacts

| Artifact | Purpose |
|---|---|
| `outputs/onnx/stage2_distill.onnx` | FP32 reference model |
| `outputs/onnx/stage2_distill_int8.onnx` | Default INT8 deployment model |
| `outputs/onnx/quantization_report.json` | Quantization parity/evaluation |
| `outputs/latency/latency_report.json` | Host-specific CPU latency report |

PyTorch training checkpoints and the STEAD dataset are intentionally not
committed. Reproduce the checkpoint with the documented Stage 1/Stage 2
pipeline; deployment and streaming can use the shipped ONNX files directly.

## Evaluation

On the 7,781-trace grouped test split (4,957 earthquake / 2,824 noise) at
detection threshold 0.80. FP is out of 2,824 noise traces, FN out of 4,957
earthquakes; each pick MAE is computed over hits only (picks within the ±500 ms
match tolerance) and carries its hit-rate:

| Model | F1 | Precision | Recall | FP (noise) | FN (eq) | P MAE (hit) | S MAE (hit) |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 ONNX | 0.9944 | 0.9910 | 0.9978 | 45 | 11 | 37.0 ms (0.981) | 71.4 ms (0.961) |
| INT8 ONNX | 0.9920 | 0.9941 | 0.9899 | 29 | 50 | 45.6 ms (0.966) | 76.2 ms (0.949) |

**Deployment-relevant false-alarm rate.** Per-window FP on curated noise is not
the metric an autonomous on-device trigger lives on — false triggers over *time*
are. At the deployed low-false-alarm operating point (threshold 0.90 + 500 ms)
the per-window noise FP rate is 11 / 2,824 (0.39 %); at the default 30 s
streaming hop (120 windows/hour) that is **≈ 0.47 false triggers/hour, ≈ 11/day**
(upper estimate; assumptions documented in
`scripts/false_alarm_rate.py` / `outputs/false_alarm/false_alarm_rate.json`).

These results apply to this curated STEAD split and preprocessing pipeline.
They do not establish performance on other regions, instruments, sample rates,
or continuous field deployments.

## Intended use

- Research and portfolio demonstrations of compact seismic inference.
- Offline or streaming candidate event detection and phase-pick generation.
- Edge-deployment experiments where outputs are reviewed or validated by an
  appropriate downstream system.

## Limitations and safety

- Not validated for earthquake early warning, public alerting, life-safety,
  emergency response, or autonomous operational decisions.
- Trained primarily on STEAD; distribution shift can materially affect results.
- Requires three ZNE channels at 100 Hz and the documented preprocessing.
- The FP32 low-false-alarm operating point has not been independently retuned
  for merged INT8 streaming output.
- The four-trace streaming demo is a path smoke test, not an accuracy study.
- Latency results are hardware-specific.

## License and attribution

Project code and shipped models are GPL-3.0-only. Training data, teacher model,
and software dependencies retain their original terms. See
`THIRD_PARTY_NOTICES.md` for citations and attribution.
