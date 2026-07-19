# Causal early-firing latency run

DATASET-GATED real STEAD run. PNW OOD remains gated; reuse scripts/pnw_zeroshot.py.

Measurement notes:
- causal_unet and sta_lta use true streaming onset-to-alarm latency.
- shipped_unet_proxy uses a right-context masking proxy, not deployable streaming.
- STA/LTA is detector-only; phase-classification metrics are N/A.
- Full STEAD training and PNW OOD evaluation are dataset-gated and were not run in smoke mode.

Causal checkpoint: /home/ubuntu/seismic-edge-picker/checkpoints/stage3_causal/best.pt
Shipped checkpoint: /home/ubuntu/seismic-edge-picker/checkpoints/stage2_distill/best.pt
STA/LTA tuned params: {'sta_s': 1.0, 'lta_s': 5.0, 'on': 3.0}

Artifacts: latency_curve.csv, summary_table.csv, recall_latency.png, run.json.
