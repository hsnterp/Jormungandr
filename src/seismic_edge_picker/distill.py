"""Stage 2 knowledge-distillation utilities (teacher = pretrained EQTransformer).

Nothing here downloads or trains on import. The heavy paths (``load_teacher`` ->
SeisBench weights, ``TeacherCache`` reads) run only when a script calls them.

Design notes
------------
- The student and EQTransformer both emit three sigmoid probability streams
  (Detection, P, S) of length 6000, so distillation is per-sample BCE between the
  student probabilities and the teacher's soft probabilities.
- Teacher outputs are cached on the DETERMINISTIC (non-augmented, ``train=False``)
  train windows. Distillation training then runs on those same windows so the
  cached soft targets stay aligned with the student's inputs. (Augmenting the
  student online would move the arrivals out from under the cached teacher stream;
  supporting that would require running the teacher online — left as a TODO.)
- Cache is CHUNKED + RESUMABLE + ATOMIC: each chunk is a self-describing ``.npz``
  written to a temp file then ``os.replace``-d into place; a JSON manifest records
  progress; a re-run skips chunks whose stored row ids already match.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .dataset import SeismicDataset
from . import splits as S


# --------------------------------------------------------------------------- #
# Teacher model
# --------------------------------------------------------------------------- #
def load_teacher(cfg, device="cpu"):
    """Load the pretrained EQTransformer teacher (same path as the baseline)."""
    import seisbench.models as sbm

    name = getattr(cfg.distill, "teacher", "EQTransformer")
    if name != "EQTransformer":
        raise ValueError(f"only EQTransformer teacher is supported, got {name!r}")
    model = sbm.EQTransformer.from_pretrained(cfg.distill.teacher_pretrained)
    model.to(device).eval()
    if model.sampling_rate != cfg.data.sampling_rate or model.in_samples != cfg.data.window_samples:
        raise ValueError(
            f"teacher expects {model.sampling_rate}Hz/{model.in_samples} samples; "
            f"config is {cfg.data.sampling_rate}Hz/{cfg.data.window_samples}"
        )
    return model


@torch.inference_mode()
def teacher_forward(model, x: torch.Tensor) -> torch.Tensor:
    """Run EQT on a (B,3,N) batch -> (B,3,N) stacked (Detection, P, S) probs."""
    out = model(x)
    return torch.stack([out[0], out[1], out[2]], dim=1)


# --------------------------------------------------------------------------- #
# Distillation loss
# --------------------------------------------------------------------------- #
def soften(prob: torch.Tensor, temperature: float, eps: float = 1e-6) -> torch.Tensor:
    """Temperature-soften a probability map via logit scaling. T==1 is a no-op."""
    if temperature == 1.0:
        return prob
    p = prob.clamp(eps, 1.0 - eps)
    logit = torch.log(p) - torch.log1p(-p)
    return torch.sigmoid(logit / temperature)


def distillation_loss(
    student, teacher, hard, stream_weights, alpha: float,
    temperature: float = 1.0, sample_weights: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
):
    """Blended hard-label + teacher-soft distillation loss.

    ``student``/``teacher``/``hard`` are (B,3,N) in [0,1]. Per element::

        L = alpha * BCE(student, softened_teacher) + (1-alpha) * BCE(student, hard)

    Reduced to a per-stream mean over (batch, time) — optionally weighting each
    trace by ``sample_weights`` (B,) — then combined as a stream-weighted mean.

    Returns ``(total, per_stream[3], (soft_scalar, hard_scalar))``.
    """
    student = student.clamp(eps, 1.0 - eps)
    target_soft = soften(teacher, temperature).clamp(eps, 1.0 - eps)
    hard = hard.clamp(eps, 1.0 - eps)

    bce_soft = F.binary_cross_entropy(student, target_soft, reduction="none")  # (B,3,N)
    bce_hard = F.binary_cross_entropy(student, hard, reduction="none")
    elem = alpha * bce_soft + (1.0 - alpha) * bce_hard                          # (B,3,N)

    per_trace_stream = elem.mean(dim=2)                                         # (B,3)
    if sample_weights is not None:
        w = sample_weights.to(per_trace_stream.dtype).clamp_min(eps)
        per_stream = (per_trace_stream * w[:, None]).sum(0) / w.sum()           # (3,)
        soft_s = ((bce_soft.mean(dim=2) * w[:, None]).sum(0) / w.sum()).mean()
        hard_s = ((bce_hard.mean(dim=2) * w[:, None]).sum(0) / w.sum()).mean()
    else:
        per_stream = per_trace_stream.mean(0)                                   # (3,)
        soft_s = bce_soft.mean()
        hard_s = bce_hard.mean()

    total = (per_stream * stream_weights).sum() / stream_weights.sum()
    return total, per_stream, (float(soft_s.detach()), float(hard_s.detach()))


def snr_weight(snr_db, lsw) -> float:
    """Monotone low-SNR upweight in [1, max_weight].

    weight = 1 at snr >= ref_db, ramps linearly up to max_weight at snr <= floor_db.
    Non-finite SNR (e.g. noise traces) -> weight 1.0.
    """
    if not getattr(lsw, "enabled", False):
        return 1.0
    if snr_db is None or not np.isfinite(snr_db):
        return 1.0
    ref, floor, mx = lsw.ref_db, lsw.floor_db, lsw.max_weight
    if snr_db >= ref:
        return 1.0
    if snr_db <= floor:
        return float(mx)
    frac = (ref - snr_db) / (ref - floor)
    return float(1.0 + frac * (mx - 1.0))


# --------------------------------------------------------------------------- #
# Chunked / resumable / atomic teacher cache
# --------------------------------------------------------------------------- #
def _atomic_np_savez(path: Path, **arrays) -> None:
    # Write to an OPEN temp handle so numpy does not append ".npz", then rename.
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def cache_signature(cfg, ordered_rows: np.ndarray) -> str:
    """Hash of the inputs that must match for a cache to be reused."""
    h = hashlib.md5()
    h.update(str(cfg.distill.teacher_pretrained).encode())
    h.update(str(cfg.data.sampling_rate).encode())
    h.update(str(cfg.data.window_samples).encode())
    h.update(str(cfg.data.preprocess.normalization).encode())
    h.update(np.asarray(ordered_rows, dtype=np.int64).tobytes())
    return h.hexdigest()


def chunk_path(cache_dir: Path, idx: int) -> Path:
    return cache_dir / f"chunk_{idx:05d}.npz"


def manifest_path(cache_dir: Path) -> Path:
    return cache_dir / "manifest.json"


def load_manifest(cache_dir: Path) -> Optional[dict]:
    p = manifest_path(cache_dir)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def chunk_is_complete(cache_dir: Path, idx: int, expected_rows: np.ndarray) -> bool:
    """A chunk counts as done iff it loads and its stored rows match exactly."""
    p = chunk_path(cache_dir, idx)
    if not p.is_file():
        return False
    try:
        with np.load(p) as z:
            rows = z["rows"]
    except (OSError, ValueError, KeyError):
        return False
    return rows.shape == expected_rows.shape and np.array_equal(rows, expected_rows)


def write_chunk(cache_dir: Path, idx: int, rows: np.ndarray,
                teacher: np.ndarray, snr: np.ndarray, dtype: str) -> None:
    _atomic_np_savez(
        chunk_path(cache_dir, idx),
        rows=np.asarray(rows, dtype=np.int64),
        teacher=np.asarray(teacher, dtype=dtype),
        snr=np.asarray(snr, dtype=np.float32),
    )


def update_manifest(cache_dir: Path, cfg, ordered_rows: np.ndarray,
                    chunk_size: int, dtype: str, completed: List[int]) -> None:
    manifest = {
        "teacher": cfg.distill.teacher,
        "teacher_pretrained": cfg.distill.teacher_pretrained,
        "signature": cache_signature(cfg, ordered_rows),
        "n_traces": int(len(ordered_rows)),
        "chunk_size": int(chunk_size),
        "n_chunks": int((len(ordered_rows) + chunk_size - 1) // chunk_size),
        "dtype": dtype,
        "stream_order": ["detection", "p", "s"],
        "window_samples": int(cfg.data.window_samples),
        "completed_chunks": sorted(int(i) for i in completed),
    }
    _atomic_write_text(manifest_path(cache_dir), json.dumps(manifest, indent=2))


def iter_chunks(n_rows: int, chunk_size: int):
    """Yield (chunk_idx, start, stop) position ranges."""
    idx = 0
    for start in range(0, n_rows, chunk_size):
        yield idx, start, min(start + chunk_size, n_rows)
        idx += 1


class TeacherCache:
    """Read side: maps a metadata row id -> cached (3,6000) fp32 teacher stream."""

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        manifest = load_manifest(self.cache_dir)
        if manifest is None:
            raise FileNotFoundError(f"no teacher cache manifest in {self.cache_dir}")
        self.manifest = manifest
        blocks, row_index = [], {}
        pos = 0
        for idx in manifest["completed_chunks"]:
            with np.load(chunk_path(self.cache_dir, idx)) as z:
                rows, teacher = z["rows"], z["teacher"]
            blocks.append(np.asarray(teacher, dtype=np.float32))
            for r in rows:
                row_index[int(r)] = pos
                pos += 1
        self.data = np.concatenate(blocks, axis=0) if blocks else np.empty((0, 3, 0))
        self.row_index = row_index

    def __contains__(self, row: int) -> bool:
        return int(row) in self.row_index

    def __len__(self) -> int:
        return len(self.row_index)

    def get(self, row: int) -> np.ndarray:
        return self.data[self.row_index[int(row)]]


class TeacherCacheDataset(Dataset):
    """Yields (x, hard_target, teacher_target, sample_weight) for distillation.

    Wraps a NON-AUGMENTED SeismicDataset over the train rows (so inputs match the
    cache) plus the on-disk teacher cache. Only rows present in the cache are
    exposed (supports capped / partial caches for cheap runs).
    """

    def __init__(self, seisbench_ds, train_rows, cfg, cache: TeacherCache):
        self.cfg = cfg
        rows = [int(r) for r in np.asarray(train_rows) if int(r) in cache]
        self.rows = np.asarray(rows, dtype=np.int64)
        # ALIGNMENT INVARIANT: train=False (hardcoded) -> deterministic, NON-
        # augmented windows identical to what cache_teacher.py fed the teacher, so
        # each cached soft target lines up sample-for-sample with the student input.
        # Never pass train=True here: augmentation would desync the pair.
        self.base = SeismicDataset(seisbench_ds, self.rows, cfg, train=False,
                                   seed=cfg.seed)
        self.cache = cache
        self.meta = seisbench_ds.metadata
        self.lsw = getattr(cfg.distill, "low_snr_weighting", None)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        x, y = self.base[i]                                  # deterministic window
        row = int(self.rows[i])
        teacher = torch.from_numpy(self.cache.get(row).copy())
        snr = S.parse_snr_db(self.meta.iloc[row].get(S.COL_SNR))
        w = snr_weight(snr, self.lsw) if self.lsw is not None else 1.0
        return x, y, teacher, torch.tensor(w, dtype=torch.float32)


def build_ordered_train_rows(cfg, cap: Optional[int] = None):
    """Return (seisbench_ds, ordered_train_rows[, snr]) with an optional cap.

    Reuses build_datasets so the split is byte-identical to Stage 1 / eval.
    """
    from .dataset import build_datasets

    datasets, split_indices = build_datasets(cfg)
    ds = datasets["train"].ds
    rows = np.asarray(split_indices["train"], dtype=np.int64)
    if cap is not None:
        rows = rows[: int(cap)]
    return ds, rows
