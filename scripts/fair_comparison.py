#!/usr/bin/env python
"""Fair student-vs-teacher comparison with two corrected-protocol fixes.

Fix 1 — threshold selection on held-out validation.
    The published chart used a fixed a-priori detection threshold (0.5) for
    both models. Here we sweep the detection threshold on the VALIDATION split
    only, fix each model's threshold at its val-max-F1 point, and then evaluate
    the TEST split ONCE at that fixed threshold. No test-set threshold tuning.

Fix 2 — EQTransformer native preprocessing.
    The published teacher was fed the STUDENT's preprocessing (demean +
    bandpass 1-45 Hz + global-std norm). EQTransformer's SeisBench 'stead'
    weights document their own conditioning (demean + per-trace PEAK norm +
    6-sample cosine taper, NO bandpass). We now feed the teacher its native
    input via the model's own ``annotate_batch_pre`` on the RAW waveform, so
    each model is scored under its own optimal conditioning. The student's
    evaluation is unchanged.

Detection rule matches evaluate.py's ``detect_event`` exactly: an event is
present at threshold t iff the detection stream has a local peak >= t, which is
equivalent to (max interior peak height) >= t. We therefore store that scalar
per trace and reproduce detect_event for every threshold from it. P/S picks use
the unchanged fixed peak_height (0.3); pick MAE is threshold-independent.

Outputs: outputs/fair_eval/comparison.json (all protocols + per-bucket AFTER).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from seismic_edge_picker.config import load_config  # noqa: E402
from seismic_edge_picker.dataset import build_datasets, _get_waveform  # noqa: E402
from seismic_edge_picker.model import build_model  # noqa: E402
from seismic_edge_picker import splits as S  # noqa: E402
from evaluate import pick_from_stream  # noqa: E402

BUCKETS = ["[0,10) dB", "[10,20) dB", "[20,100) dB"]
THR_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def det_peak(stream, distance):
    """Max interior local-peak height (reproduces detect_event for all thr)."""
    peaks, _ = find_peaks(stream, distance=distance, plateau_size=1)
    return float(stream[peaks].max()) if len(peaks) else float("-inf")


def eval_split(rows, ds, student, teacher, dev, cfg, tag):
    fs = cfg.data.sampling_rate
    dist = max(1, int(round(cfg.eval.peak_min_distance_s * fs)))
    ph = cfg.eval.peak_height
    meta = ds.metadata
    cat = meta[S.COL_CATEGORY].astype(str)
    is_eq = cat.isin(S.EARTHQUAKE_VALUES).to_numpy()

    recs = []
    B = 128
    for start in range(0, len(rows), B):
        chunk = rows[start:start + B]
        stud = torch.stack([ds[li][0] for li in chunk]).to(dev)
        raw = torch.tensor(
            np.stack([_get_waveform(ds.ds, int(ds.rows[li]), 3, cfg.data.window_samples)
                      for li in chunk]), dtype=torch.float32, device=dev)
        with torch.inference_mode():
            s_out = student(stud)                              # student pipeline
            tsp = teacher(stud)                                # teacher, student pipe
            tn = teacher(teacher.annotate_batch_pre(raw, {}))  # teacher, native pipe
        s_np = s_out.cpu().numpy()
        tsp = [o.cpu().numpy() for o in tsp]
        tn = [o.cpu().numpy() for o in tn]
        for b, li in enumerate(chunk):
            row = int(ds.rows[li])
            m = meta.iloc[row]
            gp = S.parse_scalar(m.get(S.COL_P))
            gs = S.parse_scalar(m.get(S.COL_S))
            rec = {"is_eq": bool(is_eq[row]),
                   "snr": S.parse_snr_db(m.get(S.COL_SNR)),
                   "gt_p": gp, "gt_s": gs}
            # student
            rec["s_dp"] = det_peak(s_np[b, 0], dist)
            rec["s_p"] = pick_from_stream(s_np[b, 1], ph, dist)[0]
            rec["s_s"] = pick_from_stream(s_np[b, 2], ph, dist)[0]
            # teacher student-pipe
            rec["tsp_dp"] = det_peak(tsp[0][b], dist)
            rec["tsp_p"] = pick_from_stream(tsp[1][b], ph, dist)[0]
            rec["tsp_s"] = pick_from_stream(tsp[2][b], ph, dist)[0]
            # teacher native
            rec["tn_dp"] = det_peak(tn[0][b], dist)
            rec["tn_p"] = pick_from_stream(tn[1][b], ph, dist)[0]
            rec["tn_s"] = pick_from_stream(tn[2][b], ph, dist)[0]
            recs.append(rec)
        print(f"  {tag}: {min(start + B, len(rows))}/{len(rows)}", end="\r", flush=True)
    print()
    return recs


def f1_at(recs, dp_key, thr, subset=None):
    rr = recs if subset is None else subset
    tp = fp = fn = tn_ = 0
    for r in rr:
        det = r[dp_key] >= thr
        if r["is_eq"]:
            tp += det
            fn += not det
        else:
            fp += det
            tn_ += not det
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"threshold": thr, "precision": prec, "recall": rec, "f1": f1,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn_)}


def best_threshold(val_recs, dp_key):
    sweep = [f1_at(val_recs, dp_key, float(t)) for t in THR_GRID]
    best = max(sweep, key=lambda d: (d["f1"], d["recall"]))
    return best["threshold"], best


def pick_mae_ms(recs, gt_key, pick_key, cfg, subset=None):
    """Within-tolerance pick MAE (ms) for any phase; threshold-independent."""
    fs = cfg.data.sampling_rate
    tol_ms = cfg.eval.match_tolerance_s * 1000.0
    rr = recs if subset is None else subset
    res = []
    for r in rr:
        if r["is_eq"] and np.isfinite(r[gt_key]) and r[pick_key] is not None:
            e = (r[pick_key] - r[gt_key]) / fs * 1000.0
            if abs(e) <= tol_ms:
                res.append(abs(e))
    return float(np.mean(res)) if res else None, len(res)


def bucketize(recs, edges):
    out = {b: [] for b in BUCKETS}
    for r in recs:
        snr = r["snr"]
        if not np.isfinite(snr):
            continue
        for lo, hi in zip(edges[:-1], edges[1:]):
            if lo <= snr < hi:
                lbl = f"[{lo:g},{hi:g}) dB"
                if lbl in out:
                    out[lbl].append(r)
                break
    return out


def main():
    cfg = load_config(str(REPO / "configs" / "default.yaml"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edges = list(cfg.eval.snr_buckets)

    ckpt = torch.load(REPO / "checkpoints/stage2_distill/best.pt",
                      map_location="cpu", weights_only=True)
    student = build_model(cfg).to(dev).eval()
    student.load_state_dict(ckpt["model_state_dict"])
    import seisbench.models as sbm
    teacher = sbm.EQTransformer.from_pretrained("stead").to(dev).eval()
    print(f"teacher native conditioning: demean + {teacher.norm}-norm + taper; "
          f"norm_detrend={teacher.norm_detrend}")

    datasets, _ = build_datasets(cfg)
    val = list(range(len(datasets["val"])))
    test = list(range(len(datasets["test"])))
    print(f"val={len(val)}  test={len(test)}")
    val_recs = eval_split(val, datasets["val"], student, teacher, dev, cfg, "val")
    test_recs = eval_split(test, datasets["test"], student, teacher, dev, cfg, "test")

    # ---- threshold selection on VALIDATION (fix 1) ------------------------
    s_thr, s_best = best_threshold(val_recs, "s_dp")
    tsp_thr, tsp_best = best_threshold(val_recs, "tsp_dp")
    tn_thr, tn_best = best_threshold(val_recs, "tn_dp")
    print(f"val-selected thresholds: student={s_thr:.2f} "
          f"teacher(studentpipe)={tsp_thr:.2f} teacher(native)={tn_thr:.2f}")

    def overall(recs, dp_key, thr, p_key, s_key):
        f = f1_at(recs, dp_key, thr)
        p_mae, p_n = pick_mae_ms(recs, "gt_p", p_key, cfg)
        s_mae, s_n = pick_mae_ms(recs, "gt_s", s_key, cfg)
        return {"threshold": thr, "f1": f["f1"], "precision": f["precision"],
                "recall": f["recall"], "tp": f["tp"], "fp": f["fp"],
                "fn": f["fn"], "tn": f["tn"],
                "p_mae_ms": p_mae, "p_mae_n": p_n,
                "s_mae_ms": s_mae, "s_mae_n": s_n}

    # ---- three protocols on TEST ------------------------------------------
    protocols = {
        "A_before": {
            "desc": "published protocol: fixed 0.5 threshold; teacher on student pipeline",
            "student": overall(test_recs, "s_dp", 0.5, "s_p", "s_s"),
            "teacher": overall(test_recs, "tsp_dp", 0.5, "tsp_p", "tsp_s"),
        },
        "B_fix1_only": {
            "desc": "val-selected threshold; teacher STILL on student pipeline",
            "student": overall(test_recs, "s_dp", s_thr, "s_p", "s_s"),
            "teacher": overall(test_recs, "tsp_dp", tsp_thr, "tsp_p", "tsp_s"),
        },
        "C_after": {
            "desc": "val-selected threshold + teacher NATIVE preprocessing",
            "student": overall(test_recs, "s_dp", s_thr, "s_p", "s_s"),
            "teacher": overall(test_recs, "tn_dp", tn_thr, "tn_p", "tn_s"),
        },
    }

    # ---- per-SNR-bucket for the corrected (AFTER) chart -------------------
    tb = bucketize(test_recs, edges)
    per_bucket = {}
    for b in BUCKETS:
        rr = tb[b]
        s_f = f1_at(rr, "s_dp", s_thr)["f1"]
        t_f = f1_at(rr, "tn_dp", tn_thr)["f1"]
        s_pmae, _ = pick_mae_ms(rr, "gt_p", "s_p", cfg)
        t_pmae, _ = pick_mae_ms(rr, "gt_p", "tn_p", cfg)
        s_smae, _ = pick_mae_ms(rr, "gt_s", "s_s", cfg)
        t_smae, _ = pick_mae_ms(rr, "gt_s", "tn_s", cfg)
        # also the BEFORE per-bucket for reference
        s_f0 = f1_at(rr, "s_dp", 0.5)["f1"]
        t_f0 = f1_at(rr, "tsp_dp", 0.5)["f1"]
        t_pmae0, _ = pick_mae_ms(rr, "gt_p", "tsp_p", cfg)
        t_smae0, _ = pick_mae_ms(rr, "gt_s", "tsp_s", cfg)
        per_bucket[b] = {
            "n": len(rr),
            "after": {"student_f1": s_f, "teacher_f1": t_f,
                      "student_p_mae_ms": s_pmae, "teacher_p_mae_ms": t_pmae,
                      "student_s_mae_ms": s_smae, "teacher_s_mae_ms": t_smae},
            "before": {"student_f1": s_f0, "teacher_f1": t_f0,
                       "student_p_mae_ms": s_pmae, "teacher_p_mae_ms": t_pmae0,
                       "student_s_mae_ms": s_smae, "teacher_s_mae_ms": t_smae0},
        }

    out = {
        "protocol_note": "threshold selected on val, applied once to test; "
                         "teacher scored under native SeisBench preprocessing",
        "val_selected_thresholds": {
            "student": s_thr, "teacher_studentpipe": tsp_thr, "teacher_native": tn_thr,
            "student_val_f1": s_best["f1"], "teacher_studentpipe_val_f1": tsp_best["f1"],
            "teacher_native_val_f1": tn_best["f1"],
        },
        "teacher_native_conditioning": f"demean + {teacher.norm}-norm + 6-sample cosine taper (no bandpass)",
        "student_conditioning": "demean + bandpass 1-45 Hz + global-std norm (unchanged)",
        "protocols": protocols,
        "per_bucket": per_bucket,
        "n_val": len(val), "n_test": len(test),
    }
    out_dir = REPO / "outputs" / "fair_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_dir/'comparison.json'}")

    # console before/after summary
    def line(name, p):
        s, t = p["student"], p["teacher"]
        print(f"  {name:12s} | student F1 {s['f1']:.4f}@{s['threshold']:.2f} "
              f"P {s['p_mae_ms']:.1f} S {s['s_mae_ms']:.1f}ms | teacher F1 "
              f"{t['f1']:.4f}@{t['threshold']:.2f} P {t['p_mae_ms']:.1f} "
              f"S {t['s_mae_ms']:.1f}ms")
    print("\nOVERALL (test):")
    for k, p in protocols.items():
        line(k, p)


if __name__ == "__main__":
    main()
