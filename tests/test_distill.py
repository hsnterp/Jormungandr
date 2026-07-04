"""Pure-logic tests for Stage-2 distillation utilities.

No dataset, no network: exercises the loss math, SNR weighting, and the
chunked/atomic/resumable cache with synthetic arrays only.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch

from seismic_edge_picker import distill as D
from seismic_edge_picker.losses import weighted_bce_loss


def _stream_weights():
    return torch.tensor([1.0, 5.0, 5.0])


def test_alpha_endpoints_match_plain_bce():
    torch.manual_seed(0)
    student = torch.rand(4, 3, 100)
    teacher = torch.rand(4, 3, 100)
    hard = torch.rand(4, 3, 100)
    w = _stream_weights()

    # alpha=0 -> pure hard-label BCE; alpha=1 -> pure teacher-soft BCE
    total0, _, (soft0, hard0) = D.distillation_loss(student, teacher, hard, w, alpha=0.0)
    ref_hard, _ = weighted_bce_loss(student, hard, w)
    assert torch.allclose(total0, ref_hard, atol=1e-6)

    total1, _, _ = D.distillation_loss(student, teacher, hard, w, alpha=1.0)
    ref_soft, _ = weighted_bce_loss(student, teacher, w)
    assert torch.allclose(total1, ref_soft, atol=1e-6)


def test_alpha_blend_is_convex_combo():
    torch.manual_seed(1)
    student = torch.rand(3, 3, 50)
    teacher = torch.rand(3, 3, 50)
    hard = torch.rand(3, 3, 50)
    w = _stream_weights()
    t0, _, _ = D.distillation_loss(student, teacher, hard, w, alpha=0.0)
    t1, _, _ = D.distillation_loss(student, teacher, hard, w, alpha=1.0)
    thalf, _, _ = D.distillation_loss(student, teacher, hard, w, alpha=0.5)
    assert torch.allclose(thalf, 0.5 * (t0 + t1), atol=1e-6)


def test_temperature_one_is_noop():
    p = torch.rand(2, 3, 20)
    assert torch.allclose(D.soften(p, 1.0), p)
    # softening pushes probabilities toward 0.5
    hot = D.soften(torch.full((5,), 0.9), temperature=4.0)
    assert torch.all(hot < 0.9) and torch.all(hot > 0.5)


def test_sample_weights_upweight_traces():
    # a trace with weight 10 should dominate the loss vs a weight-1 trace
    student = torch.stack([torch.full((3, 10), 0.9), torch.full((3, 10), 0.5)])
    teacher = torch.stack([torch.full((3, 10), 0.1), torch.full((3, 10), 0.5)])
    hard = teacher.clone()
    w = _stream_weights()
    sw_equal = torch.tensor([1.0, 1.0])
    sw_skew = torch.tensor([10.0, 1.0])
    eq, _, _ = D.distillation_loss(student, teacher, hard, w, 1.0, sample_weights=sw_equal)
    sk, _, _ = D.distillation_loss(student, teacher, hard, w, 1.0, sample_weights=sw_skew)
    assert sk > eq  # the bad (high-loss) trace is upweighted


def test_snr_weight_monotone():
    lsw = SimpleNamespace(enabled=True, ref_db=15.0, floor_db=0.0, max_weight=3.0)
    assert D.snr_weight(20.0, lsw) == 1.0          # high SNR -> no upweight
    assert D.snr_weight(-5.0, lsw) == 3.0          # below floor -> max
    assert D.snr_weight(float("nan"), lsw) == 1.0  # noise -> 1.0
    mid = D.snr_weight(7.5, lsw)
    assert 1.0 < mid < 3.0
    # disabled -> always 1.0
    off = SimpleNamespace(enabled=False, ref_db=15.0, floor_db=0.0, max_weight=3.0)
    assert D.snr_weight(-5.0, off) == 1.0


def _fake_cfg(cache_dir, tmp_path):
    return SimpleNamespace(
        seed=42,
        distill=SimpleNamespace(teacher="EQTransformer", teacher_pretrained="stead"),
        data=SimpleNamespace(sampling_rate=100, window_samples=60,
                             preprocess=SimpleNamespace(normalization="std")),
    )


def test_cache_write_read_roundtrip(tmp_path):
    cache_dir = tmp_path
    cfg = _fake_cfg(cache_dir, tmp_path)
    rows = np.arange(20, dtype=np.int64)
    teacher = np.random.rand(20, 3, 60).astype(np.float32)
    snr = np.linspace(-5, 25, 20).astype(np.float32)

    chunk_size = 8
    completed = []
    for idx, s, e in D.iter_chunks(len(rows), chunk_size):
        D.write_chunk(cache_dir, idx, rows[s:e], teacher[s:e], snr[s:e], "float16")
        completed.append(idx)
    D.update_manifest(cache_dir, cfg, rows, chunk_size, "float16", completed)

    cache = D.TeacherCache(cache_dir)
    assert len(cache) == 20
    for r in rows:
        got = cache.get(int(r))
        assert got.shape == (3, 60)
        # fp16 round-trip tolerance
        assert np.allclose(got, teacher[r], atol=1e-2)


def test_cache_resume_skips_completed(tmp_path):
    cache_dir = tmp_path
    cfg = _fake_cfg(cache_dir, tmp_path)
    rows = np.arange(16, dtype=np.int64)
    teacher = np.random.rand(16, 3, 60).astype(np.float32)
    snr = np.zeros(16, dtype=np.float32)
    chunk_size = 8

    # write only the first chunk
    D.write_chunk(cache_dir, 0, rows[0:8], teacher[0:8], snr[0:8], "float16")
    assert D.chunk_is_complete(cache_dir, 0, rows[0:8]) is True
    assert D.chunk_is_complete(cache_dir, 1, rows[8:16]) is False
    # wrong expected rows -> not complete (guards against row/config drift)
    assert D.chunk_is_complete(cache_dir, 0, rows[8:16]) is False


def test_atomic_write_leaves_no_tmp(tmp_path):
    cache_dir = tmp_path
    rows = np.arange(4, dtype=np.int64)
    D.write_chunk(cache_dir, 0, rows, np.zeros((4, 3, 60), np.float32),
                  np.zeros(4, np.float32), "float16")
    leftovers = [p.name for p in cache_dir.iterdir() if ".tmp." in p.name]
    assert leftovers == []
    assert D.chunk_path(cache_dir, 0).is_file()


def test_manifest_signature_changes_with_rows(tmp_path):
    cfg = _fake_cfg(tmp_path, tmp_path)
    a = D.cache_signature(cfg, np.arange(10))
    b = D.cache_signature(cfg, np.arange(11))
    assert a != b
