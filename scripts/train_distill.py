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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

SCRIPTS_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(SCRIPTS_DIR, "..", "src"))
sys.path.insert(0, SCRIPTS_DIR)

from seismic_edge_picker.config import load_config  # noqa: E402
from seismic_edge_picker.dataset import build_datasets  # noqa: E402
from seismic_edge_picker.model import build_model, count_parameters  # noqa: E402
from seismic_edge_picker.labels import build_label_mask_from_cfg  # noqa: E402
from seismic_edge_picker.preprocessing import (  # noqa: E402
    preprocess_waveform,
    preprocess_waveform_causal,
)
from seismic_edge_picker.losses import stream_weight_tensor  # noqa: E402
from seismic_edge_picker import distill as D  # noqa: E402
# reuse Stage-1 helpers so behaviour/formatting stay identical. run_epoch with
# optimizer=None is the SAME hard-label weighted-BCE validation pass Stage 1 uses.
from train import (  # noqa: E402
    build_optimizer, build_scheduler, make_loader, resolve_device, run_epoch,
    save_checkpoint, save_loss_curve,
)


CSV_FIELDS = ["epoch", "lr", "train_loss", "soft_bce", "hard_bce",
              "det_bce", "p_bce", "s_bce", "latency_loss",
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
    p.add_argument("--lr", type=float, default=None,
                   help="override distill.lr (e.g. 1e-4 for the causal net; the "
                        "default 5e-4 collapses the causal variant to constant output)")
    p.add_argument("--from-scratch", action="store_true",
                   help="skip warm-start; train the causal net from random init")
    p.add_argument("--device", default=None)
    p.add_argument("--data", action="store_true",
                   help="for --causal runs, use cached STEAD + teacher cache; otherwise synthetic smoke only")
    p.add_argument("--causal", action="store_true",
                   help="train the strictly causal SeismicUNet variant")
    p.add_argument("--lookahead", type=int, default=0,
                   help="fixed causal lookahead in samples (default: 0)")
    p.add_argument("--run-name", default=None,
                   help="override checkpoint/output run directory name")
    p.add_argument("--latency-aware", action="store_true",
                   help="optional separate causal ablation: extra weighted P-onset BCE")
    p.add_argument("--latency-weight", type=float, default=1.0)
    return p.parse_args()


def apply_causal_overrides(cfg, args) -> None:
    if args.causal:
        cfg.model.causal = True
        cfg.model.lookahead = int(args.lookahead)


def default_run_name(cfg, args, smoke: bool) -> str:
    if args.run_name:
        return args.run_name
    if args.causal:
        if args.latency_aware:
            return "stage3_causal_latency_smoke" if smoke else "stage3_causal_latency"
        return "stage3_causal_smoke" if smoke else "stage3_causal"
    return cfg.distill.smoke.run_name if smoke else cfg.distill.run_name


def default_causal_init(args) -> str | None:
    if getattr(args, "from_scratch", False):
        return None
    if args.init:
        return args.init
    if args.causal:
        candidate = Path("checkpoints/stage2_distill/best.pt")
        if candidate.is_file():
            return str(candidate)
    return None


def load_transferable_state(model, checkpoint_path: str) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    source = ckpt["model_state_dict"]
    target = model.state_dict()
    transferable = {}
    skipped = []
    for key, value in source.items():
        if key in target and target[key].shape == value.shape:
            transferable[key] = value
        else:
            got = tuple(value.shape)
            want = tuple(target[key].shape) if key in target else None
            skipped.append((key, got, want))
    missing = [key for key in target if key not in transferable]
    model.load_state_dict(transferable, strict=False)
    print(
        f"warm-start: loaded {len(transferable)}/{len(target)} tensors from "
        f"{checkpoint_path} (epoch {ckpt.get('epoch')}, val {ckpt.get('best_val_loss')})"
    )
    if skipped or missing:
        print("warm-start tensors not transferred:")
        for key, got, want in skipped[:20]:
            print(f"  skip {key}: checkpoint {got}, model {want}")
        for key in missing[:20]:
            if key not in source:
                print(f"  missing in checkpoint: {key}")
        if len(skipped) + len(missing) > 20:
            print(f"  ... {len(skipped) + len(missing) - 20} more")
    else:
        print("warm-start tensors not transferred: none (all shapes matched)")
    return ckpt


def latency_aware_p_loss(pred, hard, weight: float):
    if weight <= 0:
        return pred.new_tensor(0.0)
    target = hard[:, 1, :].clamp(0.0, 1.0)
    p = pred[:, 1, :].clamp(1e-6, 1.0 - 1e-6)
    elem = F.binary_cross_entropy(p, target, reduction="none")
    weights = 1.0 + float(weight) * target
    return (elem * weights).mean()


def synthetic_distill_tensors(cfg, n_eq: int = 8, n_noise: int = 4, seed: int = 0):
    fs = float(cfg.data.sampling_rate)
    n = int(cfg.data.window_samples)
    nch = int(cfg.data.n_channels)
    rng = np.random.default_rng(seed)

    def arrival(onset, freq, amp, tau):
        t = (np.arange(n) - onset) / fs
        env = np.where(t >= 0, np.exp(-t / tau), 0.0)
        return amp * env * np.sin(2 * np.pi * freq * t)

    xs, ys = [], []
    for _ in range(n_eq):
        p_on = int(rng.integers(int(3 * fs), int(25 * fs)))
        s_on = int(min(n - int(2 * fs), p_on + rng.integers(int(3 * fs), int(10 * fs))))
        raw = 0.15 * rng.standard_normal((nch, n)).astype(np.float32)
        for ch in range(nch):
            raw[ch] += arrival(p_on, 8.0, 2.0, 1.2)
            raw[ch] += arrival(s_on, 3.0, 3.0, 3.5)
        x = preprocess_waveform_causal(raw, cfg) if cfg.model.causal else preprocess_waveform(raw, cfg)
        y = build_label_mask_from_cfg(n, p_on, s_on, s_on + int(8 * fs), cfg)
        xs.append(x); ys.append(y)
    for _ in range(n_noise):
        raw = 0.2 * rng.standard_normal((nch, n)).astype(np.float32)
        x = preprocess_waveform_causal(raw, cfg) if cfg.model.causal else preprocess_waveform(raw, cfg)
        y = build_label_mask_from_cfg(n, None, None, None, cfg)
        xs.append(x); ys.append(y)
    order = rng.permutation(len(xs))
    x = torch.from_numpy(np.stack([xs[i] for i in order]).astype(np.float32))
    y = torch.from_numpy(np.stack([ys[i] for i in order]).astype(np.float32))
    teacher = y.clone()
    sample_weight = torch.ones(len(order), dtype=torch.float32)
    return x, y, teacher, sample_weight


def run_synthetic_smoke(args, cfg, device) -> None:
    print("no data — plumbing only")
    epochs = args.epochs or 1
    run_name = default_run_name(cfg, args, smoke=True)
    x, hard, teacher, sw = synthetic_distill_tensors(cfg, seed=cfg.seed)
    train_ds = TensorDataset(x, hard, teacher, sw)
    val_ds = TensorDataset(x, hard)
    loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    model = build_model(cfg).to(device)
    init_path = default_causal_init(args)
    if init_path:
        load_transferable_state(model, init_path)
    weights = stream_weight_tensor(cfg, device)
    optimizer = build_optimizer(cfg, model)
    smoke_lr = float(args.lr) if args.lr is not None else cfg.distill.lr
    for g in optimizer.param_groups:
        g["lr"] = smoke_lr
    scheduler = build_scheduler(cfg, optimizer, epochs)

    checkpoint_dir = Path(cfg.train.checkpoint_dir) / run_name
    log_dir = Path(cfg.train.log_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    rows_log = []
    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        total = soft_total = hard_total = latency_total = 0.0
        stream = np.zeros(3, dtype=np.float64)
        total_n = 0
        lr = optimizer.param_groups[0]["lr"]
        for xb, hb, tb, wb in loader:
            xb = xb.to(device); hb = hb.to(device); tb = tb.to(device); wb = wb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss, per_stream, (soft_s, hard_s) = D.distillation_loss(
                pred, tb, hb, weights, cfg.distill.alpha, cfg.distill.temperature, wb
            )
            latency = latency_aware_p_loss(pred, hb, args.latency_weight) if args.latency_aware else pred.new_tensor(0.0)
            loss = loss + latency
            loss.backward()
            optimizer.step()
            bn = xb.shape[0]
            total += float(loss.detach()) * bn
            soft_total += soft_s * bn
            hard_total += hard_s * bn
            latency_total += float(latency.detach()) * bn
            stream += per_stream.detach().cpu().numpy() * bn
            total_n += bn
        scheduler.step()
        stream /= total_n
        val_loss, val_stream = run_epoch(model, val_loader, weights, device)
        row = {
            "epoch": epoch, "lr": lr,
            "train_loss": total / total_n, "soft_bce": soft_total / total_n,
            "hard_bce": hard_total / total_n, "det_bce": float(stream[0]),
            "p_bce": float(stream[1]), "s_bce": float(stream[2]),
            "latency_loss": latency_total / total_n,
            "val_loss": val_loss, "val_detection_bce": float(val_stream[0]),
            "val_p_bce": float(val_stream[1]), "val_s_bce": float(val_stream[2]),
            "epoch_seconds": time.perf_counter() - t0,
        }
        rows_log.append(row)
        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, scheduler,
                        cfg, epoch, min(best_val, val_loss), row)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, scheduler,
                            cfg, epoch, best_val, row)
        print(f"synthetic epoch {epoch}/{epochs} train={row['train_loss']:.5f} val={val_loss:.5f}")
    metrics_path = log_dir / "distill_metrics.csv"
    with metrics_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader(); writer.writerows(rows_log)
    save_loss_curve(rows_log, log_dir / "distill_loss.png")
    print(f"synthetic smoke checkpoint: {checkpoint_dir / 'best.pt'}")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    apply_causal_overrides(cfg, args)
    device = resolve_device(args.device or cfg.train.device)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if args.causal and not args.data:
        run_synthetic_smoke(args, cfg, device)
        return

    dc = cfg.distill
    alpha = float(getattr(dc, "alpha", dc.blend_ratio))
    temperature = float(getattr(dc, "temperature", 1.0))
    if args.smoke:
        cache_dir = Path(args.cache_dir or (dc.cache_dir + "_smoke"))
        epochs = args.epochs or dc.smoke.epochs
        batch_size = dc.smoke.batch_size
        num_workers = dc.smoke.num_workers
        run_name = default_run_name(cfg, args, smoke=True)
    else:
        cache_dir = Path(args.cache_dir or dc.cache_dir)
        epochs = args.epochs or dc.epochs
        batch_size = cfg.train.batch_size
        num_workers = cfg.train.num_workers
        run_name = default_run_name(cfg, args, smoke=False)

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
    if args.causal:
        print(f"causal model: enabled lookahead={cfg.model.lookahead} samples; "
              "DATASET-GATED real STEAD/cache training")
    if args.latency_aware:
        print(f"latency-aware ablation: enabled weight={args.latency_weight:g} "
              "(separate experiment from primary causality run)")
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
    init_path = default_causal_init(args)
    if init_path:
        load_transferable_state(model, init_path)
    weights = stream_weight_tensor(cfg, device)
    lr = float(args.lr) if args.lr is not None else dc.lr
    optimizer = build_optimizer(cfg, model)
    # honor the distillation lr instead of Stage-1 lr
    for g in optimizer.param_groups:
        g["lr"] = lr
    scheduler = build_scheduler(cfg, optimizer, epochs)
    print(f"student params: {count_parameters(model):,}  distill lr={lr:g}"
          f"{' (from scratch)' if args.from_scratch else ''}")

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
            tot, tot_soft, tot_hard, tot_latency, tot_n = 0.0, 0.0, 0.0, 0.0, 0
            stream = np.zeros(3, dtype=np.float64)
            for x, hard, teacher, sw in loader:
                x = x.to(device); hard = hard.to(device)
                teacher = teacher.to(device); sw = sw.to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(x)
                loss, per_stream, (soft_s, hard_s) = D.distillation_loss(
                    pred, teacher, hard, weights, alpha, temperature, sw)
                latency = (
                    latency_aware_p_loss(pred, hard, args.latency_weight)
                    if args.latency_aware else pred.new_tensor(0.0)
                )
                loss = loss + latency
                loss.backward()
                if cfg.train.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                optimizer.step()
                bn = x.shape[0]
                tot += loss.item() * bn
                tot_soft += soft_s * bn
                tot_hard += hard_s * bn
                tot_latency += float(latency.detach()) * bn
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
                "latency_loss": tot_latency / tot_n,
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
                  f"[soft={row['soft_bce']:.4f} hard={row['hard_bce']:.4f} "
                  f"lat={row['latency_loss']:.4f}]  "
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
