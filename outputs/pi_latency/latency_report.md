# Phase 5c latency benchmark

- Device: CPU (`unknown`)
- Input: `(1, 3, 6000)` float32 (one 60 s window)
- Threads: 1 intra-op, 1 inter-op; ORT sequential execution
- Warmup: 20 runs per backend (excluded)
- Measured: 200 runs per backend
- Throughput: `1000 / mean latency (ms)`

| Backend | p50 (ms) | p95 (ms) | mean (ms) | windows/s |
|---|---:|---:|---:|---:|
| FP32 ONNX Runtime | 3.095 | 3.164 | 3.106 | 322.0 |
| INT8 ONNX Runtime | 2.392 | 2.400 | 2.392 | 418.1 |
