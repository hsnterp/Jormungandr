# Phase 5c latency benchmark

- Device: CPU (`AMD EPYC 7763 64-Core Processor`)
- Input: `(1, 3, 6000)` float32 (one 60 s window)
- Threads: 1 intra-op, 1 inter-op; ORT sequential execution
- Warmup: 20 runs per backend (excluded)
- Measured: 200 runs per backend
- Throughput: `1000 / mean latency (ms)`

| Backend | p50 (ms) | p95 (ms) | mean (ms) | windows/s |
|---|---:|---:|---:|---:|
| PyTorch CPU | 3.149 | 4.119 | 3.371 | 296.7 |
| FP32 ONNX Runtime | 1.543 | 1.571 | 1.567 | 638.0 |
| INT8 ONNX Runtime | 1.226 | 1.253 | 1.233 | 811.0 |
