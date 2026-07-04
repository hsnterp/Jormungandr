#!/usr/bin/env python
"""Stage 2b: distillation fine-tune of the student on cached teacher soft targets.

Reuses the Stage-1 plumbing (build_model, config, optimizer/scheduler/checkpoint
helpers from scripts/train.py) and the cached EQTransformer outputs. The loss
blends hard labels and teacher-soft targets::

    L = alpha * BCE(student, teacher_soft) + (1 - alpha) * BCE(student, hard)

Each epoch runs a HARD-LABEL validation pass on the val split (identical to
Stage-1 validation, via ``train.run_epoch``) and checkpoints / early-stops on the
validation loss — same semantics as Stage 1. Training uses the DETERMINISTIC,
non-augmented train windows so inputs stay aligned with the cached teacher outputs
(``distill.use_augmentations`` must be False; the trainer refuses otherwise).

Optionally initialised from the Stage-1 best checkpoint (recommended).

Tiny smoke (1 epoch on the smoke cache; cheap; allowed):
    python scripts/train_distill.py --config configs/default.yaml --smoke \
        --cache-dir data/teacher_cache_smoke

Full distillation (EXPENSIVE — needs the full teacher cache; do NOT run yet):
    python scripts/train_distill.py --config configs/default.yaml \
        --init checkpoints/stage1/best.pt
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

SCRIPTS_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(SCRIPTS_DIR, "..", "src"))
sys.path.insert(0, SCRIPTS_DIR)

from seismic_edge_picker.config import load_config  # noqa: E402
from seismic_edge_picker.dataset import build_datasets  # noqa: E402
from seismic_edge_picker.model import build_model, count_parameters  # noqa: E402
from seismic_edge_picker.losses import stream_weight_tensor  # noqa: E402
from seismic_edge_picker import distill as D  # noqa: E402
# reuse Stage-1 helpers so behaviour/formatting stay identical. run_epoch with
# optimizer=None is the SAME hard-label weighted-BCE validation pass Stage 1 uses.
from train import (  # noqa: E402
    build_optimizer, build_scheduler, make_loader, resolve_device, run_epoch,
    save_checkpoint, save_loss_curve,
)


CSV_FIELDS = ["epoch", "lr", "train_loss", "soft_bce", "hard_bce",
              "det_bce", "p_bce", "s_bce",
              "val_loss", "val_detection_bce", "val_p_bce", "val_s_bce",
              "epoch_seconds"]


def parse_args():
    p = argparse.ArgumentParser(description="Distillation fine-tune of the student.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--init", default=None,
                   help="student checkpoint to warm-start from (e.g. Stage-1 best.pt)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(args.device or cfg.train.device)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    dc = cfg.distill
    alpha = float(getattr(dc, "alpha", dc.blend_ratio))
    temperature = float(getattr(dc, "temperature", 1.0))
    if args.smoke:
        cache_dir = Path(args.cache_dir or (dc.cache_dir + "_smoke"))
        epochs = args.epochs or dc.smoke.epochs
        batch_size = dc.smoke.batch_size
        num_workers = dc.smoke.num_workers
        run_name = dc.smoke.run_name
    else:
        cache_dir = Path(args.cache_dir or dc.cache_dir)
        epochs = args.epochs or dc.epochs
        batch_size = cfg.train.batch_size
        num_workers = cfg.train.num_workers
        run_name = dc.run_name

    # ALIGNMENT GUARD: cached teacher outputs were produced on the deterministic,
    # non-augmented train windows. Random augmentation would move arrivals out
    # from under the cached soft targets (desync), so it is unsupported here.
    if getattr(dc, "use_augmentations", False):
        raise SystemExit(
            "distill.use_augmentations=true is UNSUPPORTED: cached teacher outputs "
            "were computed on non-augmented windows; augmenting the student would "
            "desync inputs from cached soft targets. See docs/stage2.md (future work)."
        )

    print(f"distillation: alpha={alpha} temperature={temperature} "
          f"low_snr_weighting={getattr(dc.low_snr_weighting, 'enabled', False)}  "
          f"use_augmentations=False (deterministic windows, aligned with cache)")
    print(f"cache={cache_dir}  epochs={epochs}  batch_size={batch_size}  device={device}")

    # build_datasets gives the byte-identical split used by Stage 1 / caching.
    # datasets['val'] is a train=False (non-augmented) SeismicDataset, so the val
    # pass sees exactly the same preprocessing as Stage-1 validation.
    datasets, split_indices = build_datasets(cfg)
    seisbench_ds = datasets["train"].ds
    train_rows = split_indices["train"]

    cache = D.TeacherCache(cache_dir)
    # TeacherCacheDataset forces train=False, i.e. DETERMINISTIC windows that match
    # cache_teacher.py's inputs exactly (same windowing + preprocessing, no aug).
    dataset = D.TeacherCacheDataset(seisbench_ds, train_rows, cfg, cache)
    print(f"teacher cache: {len(cache)} traces; usable train traces: {len(dataset)}")
    if len(dataset) == 0:
        raise SystemExit("no cached traces available — run cache_teacher.py first")

    val_dataset = datasets["val"]
    if args.smoke:
        val_dataset = Subset(val_dataset, range(min(dc.smoke.n_val, len(val_dataset))))
    print(f"validation traces (hard-label BCE): {len(val_dataset)}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=device.type == "cuda")
    val_loader = make_loader(val_dataset, batch_size, num_workers, False, device, cfg.seed)

    model = build_model(cfg).to(device)
    if args.init:
        ckpt = torch.load(args.init, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"warm-started student from {args.init} "
              f"(epoch {ckpt.get('epoch')}, val {ckpt.get('best_val_loss')})")
    weights = stream_weight_tensor(cfg, device)
    optimizer = build_optimizer(cfg, model)
    # honor the distillation lr instead of Stage-1 lr
    for g in optimizer.param_groups:
        g["lr"] = dc.lr
    scheduler = build_scheduler(cfg, optimizer, epochs)
    print(f"student params: {count_parameters(model):,}  distill lr={dc.lr:g}")

    checkpoint_dir = Path(cfg.train.checkpoint_dir) / run_name
    log_dir = Path(cfg.train.log_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / "distill_metrics.csv"
    curve_path = log_dir / "distill_loss.png"

    rows_log = []
    best_val = float("inf")
    stale_epochs = 0
    patience = cfg.train.early_stopping_patience  # same semantics as Stage 1
    with metrics_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            model.train()
            lr = optimizer.param_groups[0]["lr"]
            tot, tot_soft, tot_hard, tot_n = 0.0, 0.0, 0.0, 0
            stream = np.zeros(3, dtype=np.float64)
            for x, hard, teacher, sw in loader:
                x = x.to(device); hard = hard.to(device)
                teacher = teacher.to(device); sw = sw.to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(x)
                loss, per_stream, (soft_s, hard_s) = D.distillation_loss(
                    pred, teacher, hard, weights, alpha, temperature, sw)
                loss.backward()
                if cfg.train.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                optimizer.step()
                bn = x.shape[0]
                tot += loss.item() * bn
                tot_soft += soft_s * bn
                tot_hard += hard_s * bn
                stream += per_stream.detach().cpu().numpy() * bn
                tot_n += bn
            scheduler.step()
            stream /= tot_n

            # HARD-LABEL validation pass — identical to Stage-1 validation
            # (train.run_epoch with optimizer=None -> weighted_bce_loss on val).
            val_loss, val_stream = run_epoch(model, val_loader, weights, device)

            row = {
                "epoch": epoch, "lr": lr,
                "train_loss": tot / tot_n, "soft_bce": tot_soft / tot_n,
                "hard_bce": tot_hard / tot_n, "det_bce": float(stream[0]),
                "p_bce": float(stream[1]), "s_bce": float(stream[2]),
                "val_loss": val_loss, "val_detection_bce": float(val_stream[0]),
                "val_p_bce": float(val_stream[1]), "val_s_bce": float(val_stream[2]),
                "epoch_seconds": time.perf_counter() - t0,
            }
            writer.writerow(row); fh.flush(); rows_log.append(row)
            # train (distill) loss vs hard-label val loss
            plot_rows = [{"epoch": r["epoch"], "train_loss": r["train_loss"],
                          "val_loss": r["val_loss"]} for r in rows_log]
            save_loss_curve(plot_rows, curve_path)

            # checkpoint + early-stop on VALIDATION loss (Stage-1 semantics)
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                stale_epochs = 0
            else:
                stale_epochs += 1
            save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, scheduler,
                            cfg, epoch, best_val, row)
            if improved:
                save_checkpoint(checkpoint_dir / "best.pt", model, optimizer,
                                scheduler, cfg, epoch, best_val, row)
            print(f"epoch {epoch:3d}/{epochs}  lr={lr:.3g}  "
                  f"train={row['train_loss']:.5f} "
                  f"[soft={row['soft_bce']:.4f} hard={row['hard_bce']:.4f}]  "
                  f"val={val_loss:.5f} "
                  f"[det={val_stream[0]:.4f} P={val_stream[1]:.4f} S={val_stream[2]:.4f}]  "
                  f"{row['epoch_seconds']:.1f}s")

            if patience and stale_epochs >= patience:
                print(f"early stopping after {stale_epochs} stale epochs")
                break

    print(f"metrics: {metrics_path}\nloss curve: {curve_path}\n"
          f"checkpoints: {checkpoint_dir / 'best.pt'}, {checkpoint_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
