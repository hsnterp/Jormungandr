"""Grouped train/val/test splitting with NO station/event leakage.

Splitting is done on a metadata DataFrame only (no waveforms), so it is fully
unit-testable with a synthetic DataFrame. Assignment is deterministic: each
group key is hashed to a bucket in ``[0, hash_mod)`` and buckets are partitioned
by the cumulative split ratios. Because assignment is by group, every trace
sharing a group key lands in the same split.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List

import numpy as np
import pandas as pd

# SeisBench-standardized STEAD column names we rely on.
COL_CATEGORY = "trace_category"
COL_SOURCE = "source_id"
COL_STATION = "station_code"
COL_NETWORK = "station_network_code"
COL_TRACE = "trace_name"
COL_P = "trace_p_arrival_sample"
COL_S = "trace_s_arrival_sample"
COL_CODA = "trace_coda_end_sample"
COL_SNR = "trace_snr_db"

EARTHQUAKE_VALUES = ("earthquake_local", "earthquake")
NOISE_VALUES = ("noise",)


def _hash_bucket(key: str, mod: int) -> int:
    h = hashlib.md5(str(key).encode("utf-8")).hexdigest()
    return int(h, 16) % mod


def group_key(row: pd.Series, by: str = "event") -> str:
    """Compute the grouping key for a metadata row.

    ``event`` groups by source_id (so all traces of one earthquake stay
    together); noise traces (no source_id) fall back to network.station, and
    finally to the unique trace name.
    """
    if by == "event":
        src = row.get(COL_SOURCE)
        if isinstance(src, str) and src:
            return f"evt:{src}"
        if src is not None and not (isinstance(src, float) and np.isnan(src)):
            return f"evt:{src}"
    # station fallback (used for noise, and when by == "station")
    net = row.get(COL_NETWORK, "")
    sta = row.get(COL_STATION, "")
    if (isinstance(sta, str) and sta) or (isinstance(net, str) and net):
        return f"sta:{net}.{sta}"
    return f"trc:{row.get(COL_TRACE)}"


def make_splits(
    metadata: pd.DataFrame,
    ratios=(0.8, 0.1, 0.1),
    by: str = "event",
    hash_mod: int = 1000,
) -> Dict[str, np.ndarray]:
    """Return dict with ``train``/``val``/``test`` arrays of integer row indices.

    Guarantees that no group key appears in more than one split.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "split ratios must sum to 1"
    keys = metadata.apply(lambda r: group_key(r, by), axis=1)
    buckets = keys.map(lambda k: _hash_bucket(k, hash_mod)).to_numpy()

    train_hi = ratios[0] * hash_mod
    val_hi = (ratios[0] + ratios[1]) * hash_mod

    splits: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    for i, b in enumerate(buckets):
        if b < train_hi:
            splits["train"].append(i)
        elif b < val_hi:
            splits["val"].append(i)
        else:
            splits["test"].append(i)
    return {k: np.asarray(v, dtype=np.int64) for k, v in splits.items()}


def select_subset(
    metadata: pd.DataFrame,
    n_earthquake: int,
    n_noise: int,
    seed: int = 42,
) -> np.ndarray:
    """Pick a class-balanced subset of row indices (earthquake + noise).

    Sampling is deterministic given ``seed``. Returns integer row indices into
    ``metadata``.
    """
    rng = np.random.default_rng(seed)
    cat = metadata[COL_CATEGORY].astype(str)
    eq_idx = np.where(cat.isin(EARTHQUAKE_VALUES).to_numpy())[0]
    noise_idx = np.where(cat.isin(NOISE_VALUES).to_numpy())[0]

    eq_take = min(n_earthquake, len(eq_idx))
    noise_take = min(n_noise, len(noise_idx))
    eq_sel = rng.choice(eq_idx, size=eq_take, replace=False)
    noise_sel = rng.choice(noise_idx, size=noise_take, replace=False)
    out = np.concatenate([eq_sel, noise_sel])
    rng.shuffle(out)
    return out.astype(np.int64)


def parse_scalar(value) -> float:
    """Parse a possibly array-stringified numeric field to a single float.

    STEAD stores some metadata as strings like ``'[[5744.]]'`` or ``'5744.0'``.
    Returns the first finite numeric found, or NaN if none / missing.
    """
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().strip("[]")
    for tok in s.replace(",", " ").split():
        try:
            f = float(tok)
        except ValueError:
            continue
        if np.isfinite(f):
            return f
    return float("nan")


def parse_snr_db(value) -> float:
    """STEAD stores snr_db as a per-component list/string; reduce to a scalar
    (mean of finite components). Returns NaN if unparseable."""
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().strip("[]")
    parts = [p for p in s.replace(",", " ").split() if p]
    vals = []
    for p in parts:
        try:
            vals.append(float(p))
        except ValueError:
            pass
    finite = [v for v in vals if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")
