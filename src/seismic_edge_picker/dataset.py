"""SeisBench/STEAD-backed torch Dataset.

IMPORTANT: importing this module does NOT download anything. The STEAD dataset
is only touched when ``load_stead()`` (or ``build_datasets()``) is explicitly
called by a script — never at import time.

Flow per item:
    raw waveform -> demean+bandpass (filter_only) -> [train augs] -> normalize
    -> tensor;   labels rebuilt from (possibly shifted) arrival samples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from . import augment as A
from .labels import build_label_mask_from_cfg
from .preprocessing import (
    causal_filter_only,
    causal_normalize_only,
    filter_only,
    normalize,
)
from . import splits as S


def stead_cache_paths() -> tuple[Path, Path]:
    """Return the metadata and waveform paths used by SeisBench for STEAD."""
    import seisbench

    root = Path(seisbench.cache_root) / "datasets" / "stead"
    return root / "metadata.csv", root / "waveforms.hdf5"


def require_stead_cache() -> tuple[Path, Path]:
    """Refuse to instantiate STEAD unless its local cache is complete.

    This prevents a training or inspection command from accidentally starting
    the roughly 90 GB dataset download.
    """
    paths = stead_cache_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "STEAD cache is incomplete; refusing to trigger a download. "
            f"Missing: {', '.join(missing)}"
        )
    return paths


def load_stead(cfg):
    """Instantiate STEAD, optionally requiring a complete existing cache."""
    import seisbench.data as sbd

    if getattr(cfg.data, "require_cached", True):
        require_stead_cache()
    return sbd.STEAD(
        sampling_rate=cfg.data.sampling_rate,
        cache="trace",
    )


def _get_waveform(ds, row_idx: int, n_channels: int, n_samples: int) -> np.ndarray:
    """Fetch a single waveform as (n_channels, n_samples), pad/crop as needed."""
    try:
        wf = ds.get_sample(row_idx)[0]
    except AttributeError:
        wf = ds.get_waveforms(idx=row_idx)
    wf = np.asarray(wf, dtype=np.float32)
    if wf.ndim == 1:
        wf = wf[None, :]
    # channels
    if wf.shape[0] < n_channels:
        pad = np.zeros((n_channels - wf.shape[0], wf.shape[1]), dtype=np.float32)
        wf = np.concatenate([wf, pad], axis=0)
    elif wf.shape[0] > n_channels:
        wf = wf[:n_channels]
    # samples
    if wf.shape[1] < n_samples:
        pad = np.zeros((wf.shape[0], n_samples - wf.shape[1]), dtype=np.float32)
        wf = np.concatenate([wf, pad], axis=1)
    elif wf.shape[1] > n_samples:
        wf = wf[:, :n_samples]
    return wf


def _arrivals_from_meta(meta_row) -> Dict[str, Optional[float]]:
    return {
        "p": meta_row.get(S.COL_P),
        "s": meta_row.get(S.COL_S),
        "coda": meta_row.get(S.COL_CODA),
    }


class SeismicDataset(Dataset):
    """Wraps a SeisBench dataset + a list of row indices for one split."""

    def __init__(self, ds, row_indices, cfg, train: bool = False,
                 noise_pool: Optional[np.ndarray] = None, seed: int = 42):
        self.ds = ds
        self.rows = np.asarray(row_indices, dtype=np.int64)
        self.cfg = cfg
        self.train = train and cfg.augment.enabled
        self.noise_pool = noise_pool
        self.n_channels = cfg.data.n_channels
        self.n_samples = cfg.data.window_samples
        self.base_seed = seed
        self.causal_preprocessing = bool(getattr(cfg.model, "causal", False))
        self.epoch = 0
        self.metadata = ds.metadata

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        """Vary deterministic train augmentations from one epoch to the next."""
        self.epoch = int(epoch)

    def _noise_sampler(self, rng):
        """Return a callable that draws a filtered noise waveform for mixing."""
        if self.noise_pool is None or len(self.noise_pool) == 0:
            return None

        def sample():
            j = int(self.noise_pool[rng.integers(len(self.noise_pool))])
            wf = _get_waveform(self.ds, j, self.n_channels, self.n_samples)
            if self.causal_preprocessing:
                return causal_filter_only(wf, self.cfg)
            return filter_only(wf, self.cfg)

        return sample

    def __getitem__(self, i: int):
        row = int(self.rows[i])
        # Per-item/epoch RNG keeps runs reproducible while refreshing training
        # augmentations each epoch.
        rng = np.random.default_rng(self.base_seed + row + self.epoch * 1_000_003)

        wf = _get_waveform(self.ds, row, self.n_channels, self.n_samples)
        meta = self.metadata.iloc[row]
        arrivals = _arrivals_from_meta(meta)
        # STEAD stores arrival samples as array-strings (e.g. '[[5744.]]');
        # parse robustly to clean ints / None up front.
        arrivals = {
            k: (None if not np.isfinite(f := S.parse_scalar(v)) else int(round(f)))
            for k, v in arrivals.items()
        }

        x = (
            causal_filter_only(wf, self.cfg)
            if self.causal_preprocessing
            else filter_only(wf, self.cfg)
        )
        if self.train:
            x, arrivals = A.apply_augmentations(
                x, arrivals, self.cfg, rng, self._noise_sampler(rng)
            )
        if self.causal_preprocessing:
            x = causal_normalize_only(x, self.cfg)
        else:
            x = normalize(x, self.cfg.data.preprocess.normalization,
                          self.cfg.data.preprocess.eps)

        y = build_label_mask_from_cfg(
            self.n_samples, arrivals["p"], arrivals["s"], arrivals["coda"], self.cfg
        )
        return torch.from_numpy(x), torch.from_numpy(y)


def build_datasets(cfg, ds=None):
    """Load STEAD (if not provided), pick the subset, make grouped splits, and
    return ``{"train":..., "val":..., "test":...}`` of ``SeismicDataset``.

    Grouped splitting guarantees no station/event leakage across splits.
    """
    if ds is None:
        ds = load_stead(cfg)
    meta = ds.metadata

    subset = S.select_subset(
        meta, cfg.data.subset.n_earthquake, cfg.data.subset.n_noise, cfg.seed
    )
    sub_meta = meta.iloc[subset].reset_index(drop=True)
    local = S.make_splits(
        sub_meta,
        ratios=tuple(cfg.data.split.ratios),
        by=cfg.data.split.by,
        hash_mod=cfg.data.split.hash_mod,
    )
    # map local (subset) indices back to full-metadata row indices
    splits = {k: subset[v] for k, v in local.items()}

    # noise pool for mixing = noise traces in the TRAIN split
    cat = meta[S.COL_CATEGORY].astype(str)
    is_noise = cat.isin(S.NOISE_VALUES).to_numpy()
    train_noise = np.array([r for r in splits["train"] if is_noise[r]], dtype=np.int64)

    return {
        "train": SeismicDataset(ds, splits["train"], cfg, train=True,
                                noise_pool=train_noise, seed=cfg.seed),
        "val": SeismicDataset(ds, splits["val"], cfg, train=False, seed=cfg.seed),
        "test": SeismicDataset(ds, splits["test"], cfg, train=False, seed=cfg.seed),
    }, splits
