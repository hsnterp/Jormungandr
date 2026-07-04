# Stage 2 — Knowledge distillation (teacher = pretrained EQTransformer)

Distill the pretrained EQTransformer (`stead`) into the 48k-param `SeismicUNet`
student. EQTransformer is used **only** as a source of soft Detection/P/S
probability streams; the student architecture and the Stage-1 checkpoints are not
modified.

## Pipeline overview

```
Stage 2a  cache_teacher.py     EQT soft outputs on the TRAIN split  ->  data/teacher_cache/
Stage 2b  train_distill.py     student fine-tune on hard+soft blend ->  checkpoints/stage2_distill/
Stage 4   evaluate.py          same eval as Stage 1, on the student  ->  outputs/stage2_eval/
```

## Distillation loss

Student and EQTransformer both emit three sigmoid streams of length 6000, so the
soft target is a direct per-sample probability match. Per element (then reduced to
a stream-weighted mean, P/S weighted 5x like Stage 1):

```
L = alpha * BCE(student, softened_teacher) + (1 - alpha) * BCE(student, hard_label)
```

**Validation & checkpointing.** Each epoch runs a hard-label weighted-BCE
validation pass on the val split — the *same* `train.run_epoch` (with
`optimizer=None`) that Stage 1 uses — and the trainer checkpoints `best.pt` and
early-stops on the **validation loss**, matching Stage-1 semantics
(`train.early_stopping_patience`). The smoke run caps the val split to
`distill.smoke.n_val` traces so it stays tiny.

- **alpha** (`distill.alpha`, default 0.5) blends teacher-soft vs hard-label loss.
  `alpha=0` reproduces Stage-1 supervised BCE; `alpha=1` is pure distillation.
  (`distill.blend_ratio` is a legacy alias; `alpha` is authoritative.)
- **temperature** (`distill.temperature`, default 1.0 = no-op) softens the
  **teacher target only** via logit scaling `sigmoid(logit(p)/T)`. The student is
  matched at its native probabilities (the model exposes probabilities, not
  logits), so temperature is applied one-sided — documented, not classical KD.
- **low_snr_weighting** (optional, default off): a monotone per-trace multiplier
  in `[1, max_weight]` that upweights low-SNR earthquakes (weight 1.0 at/above
  `ref_db`, ramping to `max_weight` at/below `floor_db`; noise/non-finite SNR → 1).

## Teacher cache format (chunked, resumable, atomic)

- Teacher outputs are computed on the **deterministic, non-augmented** train
  windows (`train=False`), so the cached soft targets stay aligned with the
  student inputs during distillation. **This is enforced, not incidental:**
  `TeacherCacheDataset` hardcodes `train=False`, and `train_distill.py` builds its
  train windows through the same `SeismicDataset(train=False)` path
  `cache_teacher.py` used — same windowing + preprocessing, no random aug. The
  `distill.use_augmentations` flag defaults to `false`; setting it `true` makes
  the trainer **abort** (augmentation would desync inputs from the cached soft
  targets). Augmentation-aware distillation is future work (see below).
- The train rows (identical order to Stage 1's `build_datasets` split) are cut
  into fixed-size chunks (`distill.chunk_size`). Each chunk is one
  `chunk_NNNNN.npz` holding `rows` (int64 metadata ids), `teacher`
  (`(n,3,6000)` fp16), and `snr` (fp32).
- **Atomic:** each `.npz` and the `manifest.json` are written to a
  `*.tmp.<pid>` file then `os.replace`-d into place, so a kill mid-write cannot
  leave a half-written chunk.
- **Resumable:** on restart, a chunk is skipped iff its file loads *and* its
  stored `rows` match the expected ids for that position. `manifest.json` records
  a `signature` (hash of teacher weights + preprocessing-relevant config + the
  exact row list); a re-run against a changed signature aborts rather than mixing
  incompatible data.
- **Cap for cheap runs:** `distill.max_cache_samples` (or `distill.smoke.*`)
  limits how many train traces are cached.

## Commands

### Cheap / already run
```bash
# End-to-end TINY smoke: cache ~24 teacher outputs + 1 distillation epoch.
# This is the only Stage-2 run meant to execute before approval. VERIFIED GREEN.
python scripts/distill_smoke.py --config configs/default.yaml

# Pure-logic unit tests (no dataset / no network): loss blend, SNR weight,
# cache write/read/resume/atomicity.
pytest -q tests/test_distill.py
```

### Expensive / NOT yet run (require explicit approval)
```bash
# Stage 2a — FULL teacher cache over all ~59,116 train traces (GPU; large disk:
# 59,116 x 3 x 6000 x 2 bytes ~= 2.1 GB in fp16). Resumable — safe to interrupt.
python scripts/cache_teacher.py --config configs/default.yaml

# Stage 2b — FULL distillation fine-tune, warm-started from the Stage-1 best.
python scripts/train_distill.py --config configs/default.yaml \
    --init checkpoints/stage1/best.pt

# Stage 4 — evaluate the distilled student (reuses the Stage-1 evaluator).
python scripts/evaluate.py --config configs/default.yaml \
    --checkpoint checkpoints/stage2_distill/best.pt --out outputs/stage2_eval
python scripts/threshold_sweep.py --config configs/default.yaml \
    --checkpoint checkpoints/stage2_distill/best.pt --out outputs/stage2_eval
```

## Fairness caveat carried over from the baseline

The teacher is run through the project's demean + bandpass(1–45 Hz) + std pipeline
(identical inputs to the student), **not** EQTransformer's native preprocessing.
Distillation therefore transfers the teacher's behaviour *on this pipeline*; the
teacher's picks under this preprocessing are modestly worse than its published
best (P MAE 63 ms vs the student's 46 ms on the test split). Distillation is still
expected to help detection precision (the teacher's strong suit: 35 vs 415 noise
false alarms at threshold 0.5) — that is the hypothesis the full run will test.

## Status of the two Stage-2 blockers (RESOLVED 2026-07-04)

- ✅ **Validation pass + checkpoint-on-val:** implemented — a hard-label
  weighted-BCE val pass each epoch (reusing `train.run_epoch`), with best-checkpoint
  and early-stopping on val loss (Stage-1 semantics). Smoke exercises it via
  `distill.smoke.n_val`.
- ✅ **Deterministic aligned inputs:** enforced — `TeacherCacheDataset` and the
  Stage-2 train loop use the same `SeismicDataset(train=False)` windows as
  `cache_teacher.py`; `distill.use_augmentations` defaults `false` and the trainer
  aborts if it is `true`.

## Future enhancements (NOT part of the first Stage 2 run)

- **Augmentation-aware distillation:** to reintroduce Stage-1's window-shift /
  noise-mix / channel-dropout without desyncing the teacher targets, either
  (a) run the teacher **online** per augmented batch (no cache; expensive), or
  (b) apply the **same geometric augmentation to the cached teacher stream** as to
  the input. Deferred; gated behind `distill.use_augmentations`, which is
  unsupported until one of these is built.
- **Cache memory:** `TeacherCache` loads all chunks into RAM (~2.1 GB at full
  scale). Fine for a GPU box; switch to a memmap if memory-constrained.
- **Hyperparameters:** `alpha`, `temperature`, `low_snr_weighting`, `epochs`,
  `lr` are unswept defaults; tune after the first full run.
