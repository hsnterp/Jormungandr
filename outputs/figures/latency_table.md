## Raspberry Pi inference latency

Stage 2 distilled model, single 60 s window (`(1, 3, 6000)` float32), 1 intra-op / 1 inter-op thread, ONNX Runtime 1.27.0 sequential execution; 20 warmup + 200 measured runs. Throughput = 1000 / mean latency.

**Device:** Raspberry Pi 5 (Broadcom BCM2712, Arm Cortex-A76 @ aarch64), 4 cores, Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41.

| Backend | p50 (ms) | p95 (ms) | mean (ms) | throughput (windows/s) |
|---|---:|---:|---:|---:|
| FP32 ONNX Runtime | 3.095 | 3.164 | 3.106 | 322.0 |
| INT8 ONNX Runtime | 2.392 | 2.400 | 2.392 | 418.1 |

On the Pi, INT8 ONNX Runtime runs a 60 s window in **2.39 ms p50 (418 windows/s)** — 1.30× faster than FP32 and about 12,543× real-time headroom against the 30 s streaming hop, confirming real-time edge inference on commodity Arm hardware.
