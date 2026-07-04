#!/usr/bin/env python
"""Stage 2a: cache pretrained-EQTransformer soft outputs for the TRAIN split.

Chunked, resumable, atomic. Runs the teacher on the DETERMINISTIC (non-augmented)
train windows and stores (Detection, P, S) probability streams per trace, keyed by
metadata row id. Re-running skips chunks already on disk whose row ids match, so an
interrupted cache can be restarted safely.

Cheap smoke (a couple dozen traces, honors distill.smoke.max_cache_samples):
    python scripts/cache_teacher.py --config configs/default.yaml --smoke

Full cache (EXPENSIVE — ~59k traces; do NOT run without approval):
    python scripts/cache_teacher.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from seismic_edge_picker.config import load_config  # noqa: E402
from seismic_edge_picker.dataset import SeismicDataset  # noqa: E402
from seismic_edge_picker import distill as D  # noqa: E402
from seismic_edge_picker import splits as S  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Cache EQTransformer teacher outputs.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--smoke", action="store_true",
                   help="tiny cache using distill.smoke.* (cheap; allowed)")
    p.add_argument("--cache-dir", default=None, help="override distill.cache_dir")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.smoke:
        cap = cfg.distill.smoke.max_cache_samples
        chunk_size = cfg.distill.smoke.chunk_size
        cache_dir = Path(args.cache_dir or (cfg.distill.cache_dir + "_smoke"))
    else:
        cap = cfg.distill.max_cache_samples
        chunk_size = cfg.distill.chunk_size
        cache_dir = Path(args.cache_dir or cfg.distill.cache_dir)
    dtype = cfg.distill.cache_dtype
    cache_dir.mkdir(parents=True, exist_ok=True)

    ds, ordered_rows = D.build_ordered_train_rows(cfg, cap=cap)
    n = len(ordered_rows)
    n_chunks = (n + chunk_size - 1) // chunk_size
    print(f"teacher cache -> {cache_dir}")
    print(f"traces={n}  chunk_size={chunk_size}  chunks={n_chunks}  dtype={dtype}  "
          f"device={device}  mode={'SMOKE' if args.smoke else 'FULL'}")

    # signature guard: refuse to mix incompatible runs in one dir
    manifest = D.load_manifest(cache_dir)
    sig = D.cache_signature(cfg, ordered_rows)
    if manifest is not None and manifest.get("signature") not in (None, sig):
        raise SystemExit(
            f"existing manifest signature != current (rows/config changed). "
            f"Use a fresh --cache-dir or clear {cache_dir}."
        )

    base = SeismicDataset(ds, ordered_rows, cfg, train=False, seed=cfg.seed)
    meta = ds.metadata
    teacher = D.load_teacher(cfg, device)
    print(f"teacher params: {sum(p.numel() for p in teacher.parameters()):,}")

    completed = list(manifest["completed_chunks"]) if manifest else []
    t0 = time.perf_counter()
    done_traces = 0
    for idx, start, stop in D.iter_chunks(n, chunk_size):
        expected_rows = ordered_rows[start:stop]
        if D.chunk_is_complete(cache_dir, idx, expected_rows):
            if idx not in completed:
                completed.append(idx)
            done_traces += len(expected_rows)
            continue

        loader = DataLoader(
            torch.utils.data.Subset(base, range(start, stop)),
            batch_size=args.batch_size, shuffle=False, num_workers=0,
        )
        outs = []
        with torch.inference_mode():
            for x, _ in loader:
                x = x.to(device, non_blocking=True)
                outs.append(D.teacher_forward(teacher, x).cpu().numpy())
        teacher_arr = np.concatenate(outs, axis=0)  # (chunk, 3, 6000)
        snr = np.array(
            [S.parse_snr_db(meta.iloc[int(r)].get(S.COL_SNR)) for r in expected_rows],
            dtype=np.float32,
        )
        D.write_chunk(cache_dir, idx, expected_rows, teacher_arr, snr, dtype)
        completed.append(idx)
        D.update_manifest(cache_dir, cfg, ordered_rows, chunk_size, dtype, completed)
        done_traces += len(expected_rows)
        elapsed = time.perf_counter() - t0
        rate = done_traces / elapsed if elapsed else 0.0
        print(f"chunk {idx + 1}/{n_chunks} written ({len(expected_rows)} traces)  "
              f"{done_traces}/{n} done  {rate:.0f} tr/s")

    D.update_manifest(cache_dir, cfg, ordered_rows, chunk_size, dtype, completed)
    print(f"cache complete: {len(set(completed))}/{n_chunks} chunks, {n} traces, "
          f"{time.perf_counter() - t0:.1f}s")
    print(f"manifest: {D.manifest_path(cache_dir)}")


if __name__ == "__main__":
    main()
