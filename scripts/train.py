#!/usr/bin/env python
"""Stage 1 supervised training for SeismicUNet on cached STEAD data.

Full training:
    python scripts/train.py --config configs/default.yaml

One-epoch real-data smoke test:
    python scripts/train.py --config configs/default.yaml --smoke-test
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from seismic_edge_picker.config import config_to_dict, load_config  # noqa: E402
from seismic_edge_picker.dataset import build_datasets, require_stead_cache  # noqa: E402
from seismic_edge_picker.losses import stream_weight_tensor, weighted_bce_loss  # noqa: E402
from seismic_edge_picker.model import build_model, count_parameters  # noqa: E402


CSV_FIELDS = [
    "epoch",
    "lr",
    "train_loss",
    "train_detection_bce",
    "train_p_bce",
    "train_s_bce",
    "val_loss",
    "val_detection_bce",
    "val_p_bce",
    "val_s_bce",
    "epoch_seconds",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SeismicUNet with supervised weighted BCE on cached STEAD."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run one epoch on the tiny real-data subset configured under train.smoke_test",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="override train.epochs for non-smoke runs",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but unavailable; using CPU")
        return torch.device("cpu")
    return torch.device(requested)


def build_optimizer(cfg, model):
    name = cfg.train.optimizer.lower()
    kwargs = {
        "lr": cfg.train.lr,
        "weight_decay": cfg.train.weight_decay,
    }
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    raise ValueError(f"unsupported optimizer: {cfg.train.optimizer!r}")


def build_scheduler(cfg, optimizer, epochs: int):
    name = cfg.train.scheduler.lower()
    if name != "cosine":
        raise ValueError(f"unsupported scheduler: {cfg.train.scheduler!r}")
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=cfg.train.min_lr,
    )


def make_loader(dataset, batch_size, num_workers, shuffle, device, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def run_epoch(model, loader, weights, device, optimizer=None, grad_clip=None):
    training = optimizer is not None
    model.train(training)
    total_samples = 0
    total_loss = 0.0
    stream_loss = np.zeros(3, dtype=np.float64)

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for x, target in loader:
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss, per_stream = weighted_bce_loss(pred, target, weights)
            if training:
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            n = x.shape[0]
            total_samples += n
            total_loss += loss.item() * n
            stream_loss += per_stream.detach().cpu().numpy() * n

    if total_samples == 0:
        raise RuntimeError("data loader is empty")
    return total_loss / total_samples, stream_loss / total_samples


def save_checkpoint(path, model, optimizer, scheduler, cfg, epoch, best_val, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val,
            "metrics": row,
            "config": config_to_dict(cfg),
        },
        tmp_path,
    )
    os.replace(tmp_path, path)


def save_loss_curve(rows, path):
    epochs = [row["epoch"] for row in rows]
    train = [row["train_loss"] for row in rows]
    val = [row["val_loss"] for row in rows]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train, marker="o", label="train weighted BCE")
    ax.plot(epochs, val, marker="o", label="validation weighted BCE")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Stage 1 supervised training")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("--epochs must be at least 1")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    smoke = args.smoke_test
    smoke_cfg = cfg.train.smoke_test
    epochs = smoke_cfg.epochs if smoke else (args.epochs or cfg.train.epochs)
    batch_size = smoke_cfg.batch_size if smoke else cfg.train.batch_size
    num_workers = smoke_cfg.num_workers if smoke else cfg.train.num_workers
    run_name = smoke_cfg.run_name if smoke else cfg.train.run_name
    device = resolve_device(cfg.train.device)

    metadata_path, waveform_path = require_stead_cache()
    print(f"STEAD cache: {metadata_path.parent}")
    print(
        f"mode={'SMOKE' if smoke else 'FULL'}  device={device}  epochs={epochs}  "
        f"batch_size={batch_size}  workers={num_workers}"
    )

    datasets, split_indices = build_datasets(cfg)
    train_dataset = datasets["train"]
    val_dataset = datasets["val"]
    if smoke:
        train_dataset = Subset(
            train_dataset, range(min(smoke_cfg.n_train, len(train_dataset)))
        )
        val_dataset = Subset(
            val_dataset, range(min(smoke_cfg.n_val, len(val_dataset)))
        )

    print(
        f"split sizes: train={len(split_indices['train'])} "
        f"val={len(split_indices['val'])} test={len(split_indices['test'])}"
    )
    print(f"run sizes: train={len(train_dataset)} val={len(val_dataset)}")
    print(
        f"cache files: metadata={metadata_path.stat().st_size / 1e6:.1f} MB "
        f"waveforms={waveform_path.stat().st_size / 1e9:.1f} GB"
    )

    train_loader = make_loader(
        train_dataset, batch_size, num_workers, True, device, cfg.seed
    )
    val_loader = make_loader(
        val_dataset, batch_size, num_workers, False, device, cfg.seed
    )

    model = build_model(cfg).to(device)
    weights = stream_weight_tensor(cfg, device)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, epochs)
    print(f"model parameters: {count_parameters(model):,}")
    print(
        f"optimizer={cfg.train.optimizer} lr={cfg.train.lr:g} "
        f"scheduler={cfg.train.scheduler} min_lr={cfg.train.min_lr:g}"
    )

    checkpoint_dir = Path(cfg.train.checkpoint_dir) / run_name
    log_dir = Path(cfg.train.log_dir) / run_name
    metrics_path = log_dir / cfg.train.metrics_filename
    curve_path = log_dir / cfg.train.loss_curve_filename
    log_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    best_val = float("inf")
    stale_epochs = 0
    patience = cfg.train.early_stopping_patience
    run_start = time.perf_counter()

    with metrics_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        csv_file.flush()

        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            datasets["train"].set_epoch(epoch - 1)
            lr = optimizer.param_groups[0]["lr"]
            train_loss, train_stream = run_epoch(
                model,
                train_loader,
                weights,
                device,
                optimizer=optimizer,
                grad_clip=cfg.train.grad_clip,
            )
            val_loss, val_stream = run_epoch(model, val_loader, weights, device)
            scheduler.step()

            row = {
                "epoch": epoch,
                "lr": lr,
                "train_loss": train_loss,
                "train_detection_bce": float(train_stream[0]),
                "train_p_bce": float(train_stream[1]),
                "train_s_bce": float(train_stream[2]),
                "val_loss": val_loss,
                "val_detection_bce": float(val_stream[0]),
                "val_p_bce": float(val_stream[1]),
                "val_s_bce": float(val_stream[2]),
                "epoch_seconds": time.perf_counter() - epoch_start,
            }
            writer.writerow(row)
            csv_file.flush()
            rows.append(row)

            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                stale_epochs = 0
            else:
                stale_epochs += 1

            save_checkpoint(
                checkpoint_dir / "last.pt",
                model,
                optimizer,
                scheduler,
                cfg,
                epoch,
                best_val,
                row,
            )
            if improved:
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    cfg,
                    epoch,
                    best_val,
                    row,
                )
            save_loss_curve(rows, curve_path)

            print(
                f"epoch {epoch:3d}/{epochs}  lr={lr:.3g}  "
                f"train={train_loss:.5f} "
                f"[det={train_stream[0]:.4f} P={train_stream[1]:.4f} "
                f"S={train_stream[2]:.4f}]  "
                f"val={val_loss:.5f} "
                f"[det={val_stream[0]:.4f} P={val_stream[1]:.4f} "
                f"S={val_stream[2]:.4f}]  "
                f"{row['epoch_seconds']:.1f}s"
            )

            if patience and stale_epochs >= patience:
                print(f"early stopping after {stale_epochs} stale epochs")
                break

    print(f"finished in {time.perf_counter() - run_start:.1f}s")
    print(f"metrics: {metrics_path}")
    print(f"loss curve: {curve_path}")
    print(f"checkpoints: {checkpoint_dir / 'best.pt'}, {checkpoint_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
