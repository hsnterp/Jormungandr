#!/usr/bin/env python
"""Render the minimal 'STEAD data vs STA/LTA vs model' comparison as an MP4.

No axes, no titles, no byte counters -- three scrolling signal lanes:

    top     the real STEAD waveform being fed into both detectors, labeled
            with the dataset's own trace id, plus dashed P/S lines at the
            catalog arrival times (a small dot marks where the model itself
            placed each pick)
    middle  STA/LTA ratio, with its fixed trigger line
    bottom  our model's detection probability, with its fixed trigger line

The lower two lanes wash a brief color tint under the line when their gate
fires, so the two rows visibly light up at different rates on the same real
STEAD earthquakes (STA/LTA flickers more; the model stays quiet except on
genuine picks). The P/S lines span all three lanes so it's visible how the
gates react relative to the true phase arrivals.

Each of the 4 real STEAD earthquakes plays as its own independent segment,
followed by a 5th segment: the pure-noise trace, where neither gate should
fire -- the quiet case. Segments are joined with a short dip-to-black (fade
out, brief hold, fade in) instead of a hard cut. Nothing else bleeds across
segments -- each starts from a blank lane. Pure numpy/OpenCV rendering --
no matplotlib, no ffmpeg.

Reads outputs/demo/gating_traces.json (written by export_gating_traces.py).
Run that first if it's missing.

    python3 scripts/export_gating_video.py
    open outputs/demo/gating_minimal.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO / "outputs" / "demo"

BG = np.array([11, 12, 14], dtype=np.float64)           # near-black
LINE = np.array([232, 232, 236], dtype=np.float64)      # near-white signal line
THRESH_LINE = np.array([90, 92, 98], dtype=np.float64)  # dim dashed threshold
DIVIDER = np.array([40, 42, 46], dtype=np.float64)
LABEL = (140, 140, 140)
TINT_STALTA = np.array([70, 150, 255], dtype=np.float64)   # amber accent (BGR) -- fired = sent
TINT_MODEL = np.array([180, 220, 90], dtype=np.float64)    # green accent (BGR) -- fired = sent
SUBLABEL = (100, 100, 100)
P_COLOR = (255, 190, 80)     # light blue (BGR) -- catalog P arrival
S_COLOR = (200, 120, 255)    # magenta (BGR)    -- catalog S arrival
MODEL_PICK = (120, 255, 120)  # bright green -- where the model placed its own pick

WINDOW_S = 20.0        # seconds of history visible at once
HIGHLIGHT_S = 10.0     # fire flash fades out over this long (matches packet clip length)
TAIL_S = 8.0           # extra settle time after a trace ends, before the transition
FADE_S = 0.4           # dip-to-black fade in/out at each segment's edges
HOLD_S = 0.3           # pure-black hold between segments


def load_bundle(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(
            f"{path} not found -- run `python3 scripts/export_gating_traces.py` first."
        )
    return json.loads(path.read_text())


def lane_pixels(values, t_src, t_query, vmax):
    """Interpolate a signal onto the query time axis, mapped to [0,1] lane height.
    Queries outside the trace's own local timeline fall back to 0 (blank) --
    this is what makes each trace self-contained with no bleed from neighbors."""
    v = np.interp(t_query, t_src, values, left=0.0, right=0.0)
    return np.clip(v / vmax, 0.0, 1.0)


def fire_alpha(fires, gate, t_query):
    alpha = np.zeros_like(t_query)
    for t_fire, g in fires:
        if g != gate:
            continue
        d = t_query - t_fire
        a = np.clip(1.0 - d / HIGHLIGHT_S, 0.0, 1.0)
        a[d < 0] = 0.0
        alpha = np.maximum(alpha, a)
    return alpha


def draw_wave_lane(canvas, y0, y1, x, values, label, subtitle=""):
    h = y1 - y0
    mid = (y0 + y1) // 2
    ys = (mid - values * (h / 2 * 0.85)).astype(np.int32)
    pts = np.stack([x.astype(np.int32), ys], axis=1).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], False, tuple(float(c) for c in LINE), 1, cv2.LINE_AA)
    cv2.putText(canvas, label, (14, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                LABEL, 1, cv2.LINE_AA)
    if subtitle:
        (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(canvas, subtitle, (14 + lw + 16, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    SUBLABEL, 1, cv2.LINE_AA)


def _time_to_x(t_val, playhead, width):
    if t_val is None:
        return None
    xf = (t_val - (playhead - WINDOW_S)) / WINDOW_S * width
    if -1 <= xf <= width + 1:
        return int(round(xf))
    return None


def draw_pick_line(canvas, x_pix, color, label, height):
    for y in range(0, height, 10):
        cv2.line(canvas, (x_pix, y), (x_pix, min(y + 5, height - 1)), color, 1, cv2.LINE_AA)
    cv2.putText(canvas, label, (x_pix + 5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                color, 1, cv2.LINE_AA)


def draw_gate_lane(canvas, y0, y1, x, values01, tint_alpha, tint_color, thresh01, label):
    h = y1 - y0
    w = canvas.shape[1]

    if tint_alpha.max() > 0:
        strength = (tint_alpha[None, :, None] * 0.35)
        canvas[y0:y1, :, :] = canvas[y0:y1, :, :] * (1 - strength) + tint_color * strength

    ty = int(y1 - thresh01 * h)
    thresh_color = tuple(float(c) for c in THRESH_LINE)
    for xs in range(0, w, 14):
        cv2.line(canvas, (xs, ty), (min(xs + 7, w - 1), ty), thresh_color, 1, cv2.LINE_AA)

    ys = (y1 - values01 * h).astype(np.int32)
    pts = np.stack([x.astype(np.int32), ys], axis=1).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], False, tuple(float(c) for c in LINE), 2, cv2.LINE_AA)

    cv2.putText(canvas, label, (14, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                LABEL, 1, cv2.LINE_AA)


def render_trace(writer, tr, meta, width, height, fps, speed):
    t = np.asarray(tr["t"], dtype=np.float64)
    wave = np.asarray(tr["wave"][0], dtype=np.float64)  # single channel, already ~[-1, 1]
    ratio = np.asarray(tr["ratio"], dtype=np.float64)
    det = np.asarray(tr["det"], dtype=np.float64)
    fires = [(f["t"], f["gate"]) for f in tr["fires"]]
    catalog_p, catalog_s = tr["catalog"]["p"], tr["catalog"]["s"]
    model_p, model_s = tr["model"]["p"], tr["model"]["s"]

    stalta_on = meta["stalta_on"]
    det_thresh = meta["det_threshold"]

    x = np.arange(width, dtype=np.float64)
    n_local_frames = int((tr["n_seconds"] + TAIL_S) / speed * fps)
    fade_frames = max(1, int(FADE_S * fps))

    h3 = height // 3
    pad = 10

    for i in range(n_local_frames):
        playhead = i * speed / fps
        t_query = playhead - WINDOW_S + (x / width) * WINDOW_S

        canvas = np.tile(BG, (height, width, 1)).astype(np.float32)
        cv2.line(canvas, (0, h3), (width, h3), tuple(float(c) for c in DIVIDER), 1, cv2.LINE_AA)
        cv2.line(canvas, (0, 2 * h3), (width, 2 * h3), tuple(float(c) for c in DIVIDER), 1, cv2.LINE_AA)

        wave_vals = np.interp(t_query, t, wave, left=0.0, right=0.0)
        draw_wave_lane(canvas, pad, h3 - pad, x, wave_vals, "STEAD DATA", tr["id"])

        top_vals = lane_pixels(ratio, t, t_query, vmax=10.0)
        top_alpha = fire_alpha(fires, "stalta", t_query)
        draw_gate_lane(canvas, h3 + pad, 2 * h3 - pad, x, top_vals, top_alpha, TINT_STALTA,
                       stalta_on / 10.0, "STA/LTA")

        bot_vals = lane_pixels(det, t, t_query, vmax=1.0)
        bot_alpha = fire_alpha(fires, "model", t_query)
        draw_gate_lane(canvas, 2 * h3 + pad, height - pad, x, bot_vals, bot_alpha, TINT_MODEL,
                       det_thresh, "MODEL")

        # ground-truth P/S arrivals, spanning all three lanes so their timing
        # against the gate lanes below is directly visible
        xp = _time_to_x(catalog_p, playhead, width)
        xs = _time_to_x(catalog_s, playhead, width)
        if xp is not None:
            draw_pick_line(canvas, xp, P_COLOR, "P", height)
        if xs is not None:
            draw_pick_line(canvas, xs, S_COLOR, "S", height)

        # the model's own recovered picks -- a small dot at the STEAD lane
        # floor, offset from the dashed catalog line by however close it got
        xmp = _time_to_x(model_p, playhead, width)
        xms = _time_to_x(model_s, playhead, width)
        if xmp is not None:
            cv2.circle(canvas, (xmp, h3 - 16), 3, MODEL_PICK, -1, cv2.LINE_AA)
        if xms is not None:
            cv2.circle(canvas, (xms, h3 - 16), 3, MODEL_PICK, -1, cv2.LINE_AA)

        fade = 1.0
        if i < fade_frames:
            fade = (i + 1) / fade_frames
        elif i >= n_local_frames - fade_frames:
            fade = (n_local_frames - i) / fade_frames
        fade = max(0.0, min(1.0, fade))

        writer.write((canvas * fade).astype(np.uint8))

    return n_local_frames


def render(data, out_path: Path, width=1280, height=720, fps=30, speed=6.0, quakes_only=False):
    meta = data["meta"]
    traces = [tr for tr in data["traces"] if (tr["is_eq"] or not quakes_only)]

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))
    black = np.zeros((height, width, 3), dtype=np.uint8)
    hold_frames = max(1, int(HOLD_S * fps))

    total_frames = 0
    for k, tr in enumerate(traces):
        total_frames += render_trace(writer, tr, meta, width, height, fps, speed)
        if k < len(traces) - 1:
            for _ in range(hold_frames):
                writer.write(black)
            total_frames += hold_frames
    writer.release()
    return len(traces), total_frames, total_frames / fps


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", default=str(DEMO_DIR / "gating_traces.json"))
    ap.add_argument("--out", default=str(DEMO_DIR / "gating_minimal.mp4"))
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--speed", type=float, default=6.0, help="playback speed-up factor")
    ap.add_argument("--quakes-only", action="store_true",
                     help="drop the pure-noise 5th segment and render only the 4 earthquakes")
    args = ap.parse_args()

    data = load_bundle(Path(args.traces))
    n_traces, n_frames, secs = render(data, Path(args.out), args.width, args.height,
                                       args.fps, args.speed, args.quakes_only)
    print(f"[video] wrote {args.out}  ({n_traces} events, {n_frames} frames, "
          f"{secs:.1f}s @ {args.fps}fps)")


if __name__ == "__main__":
    main()
