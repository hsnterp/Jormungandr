# Causal early-firing latency run

Synthetic smoke only; no STEAD training/eval and no PNW OOD eval were run.

Measurement notes:
- causal_unet and sta_lta use true streaming onset-to-alarm latency.
- shipped_unet_proxy uses a right-context masking proxy, not deployable streaming.
- STA/LTA is detector-only; phase-classification metrics are N/A.
- Full STEAD training and PNW OOD evaluation are dataset-gated and were not run in smoke mode.

Causal checkpoint: /home/ubuntu/seismic-edge-picker/checkpoints/stage3_causal_smoke/best.pt
Shipped checkpoint: /home/ubuntu/seismic-edge-picker/checkpoints/stage2_distill/best.pt
STA/LTA tuned params: {'sta_s': 0.2, 'lta_s': 3.0, 'on': 2.0}

Artifacts: latency_curve.csv, summary_table.csv, recall_latency.png, run.json.
