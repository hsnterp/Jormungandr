#!/usr/bin/env python
"""Live edge demo: transmission-gating a bandwidth-limited seismic sensor.

THESIS
------
A remote, bandwidth-limited sensor should not stream everything it digitizes; it
should transmit only when something worth reporting happens. Two on-device
triggers are compared as the "gate": a classic STA/LTA energy detector and this
project's causal INT8 U-Net. The bundled replay tests whether the model can avoid
some energy-only triggers while retaining the known events. The headline is
measured bytes on the wire, with identical encoding for all paths.

HARDWARE HONESTY (read before recording)
----------------------------------------
The MPU6050 is a consumer MEMS accelerometer. The MPU3050 is a gyroscope and
measures angular rate, not seismic acceleration. Neither validates earthquake
detection: live sensor modes demonstrate physical acquisition, causal inference,
and transmission gating only. Genuine P/S behavior is shown only by `--source
replay`, using real STEAD waveforms. The console reports what actually fired; no
tap-vs-shake or field false-alarm claim is inferred from a live MEMS session.

SOURCES
-------
  --source smoke   deterministic synthetic accel baseline + injected transients;
                   no hardware, no dataset. Both gates exercised; counters diverge.
  --source mpu6050 live I2C acceleration from an MPU6050 (smbus2).
  --source mpu3050 live I2C angular rate from an MPU3050 (smbus2).
                   Hardware modes fail clearly by default; --fallback-smoke opts
                   into a synthetic fallback for presentation resilience.
  --source replay  streams outputs/demo/replay_traces.npz (REAL STEAD earthquakes,
                   verified pickable) through the SAME causal pipeline -- genuine
                   P/S picks that the live motion sensors cannot produce. Labeled REPLAY.

Six stages: (1) source -> raw (3,N) @100Hz; (2) causal preprocess + causal INT8
inference, plus STA/LTA, both gates each hop; (3) transmission gating + byte
accounting; (4) console (default) or --plot view; (5) replay; (6) session end ->
outputs/demo/bytes_report.json + outputs/demo/bytes_comparison.png (headline).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from seismic_edge_picker.config import load_config  # noqa: E402
from seismic_edge_picker.preprocessing import CausalPreprocessor  # noqa: E402
from seismic_edge_picker.streaming import (  # noqa: E402
    causal_stream_probabilities,
    extract_phase_picks,
)
from signal_utils import arrival_wave, sta_lta_ratio  # noqa: E402
from false_alarm_rate import make_session  # noqa: E402

# --- fixed knobs (documented) -----------------------------------------------
BYTES_PER_SAMPLE = 2       # 16-bit samples; the RATIO is the point, not the abs
CLIP_PRE_S = 3.0           # triggering clip: 3 s pre-trigger ...
CLIP_POST_S = 7.0          # ... + 7 s post = a 10 s event packet
STA_S, LTA_S, STA_LTA_ON = 1.0, 5.0, 3.0   # tuned STA/LTA (see outputs/causal/run.json)
DET_THRESHOLD = 0.80       # model detection gate (deploy.streaming.detection_threshold)
P_THRESHOLD = 0.30
S_THRESHOLD = 0.30
HOP_S = 0.5                # streaming hop (seconds)
REFRACTORY_S = 12.0        # trigger cooldown: at most one packet per event/gate
DEMO_DIR = REPO / "outputs" / "demo"


# ---------------------------------------------------------------------------
# Trigger + packet accounting
# ---------------------------------------------------------------------------
class EdgeTrigger:
    """Fires once on a rising edge (value crosses `on` from below); re-arms only
    after the value falls below `off` (hysteresis) AND a refractory cooldown has
    elapsed since the last fire -- so one sustained event = one transmission."""

    def __init__(self, on: float, off: float | None = None, refractory_s: float = 0.0):
        self.on = on
        self.off = on if off is None else off
        self.refractory_s = refractory_s
        self.armed = False
        self.last_fire_t = -1e18

    def update(self, value: float, t_s: float) -> bool:
        if value < self.off:
            self.armed = False
        if value >= self.on and not self.armed and (t_s - self.last_fire_t) >= self.refractory_s:
            self.armed = True
            self.last_fire_t = t_s
            return True
        if value >= self.on:
            self.armed = True  # above-threshold but suppressed (refractory/held)
        return False

    def reset(self):
        self.armed = False
        self.last_fire_t = -1e18


def serialize_packet(
    clip: np.ndarray,
    picks: dict,
    seg: str,
    t_s: float,
    *,
    fs: float = 100.0,
    gate: str = "event",
) -> int:
    """Serialize ONE transmission packet the same way every time; return its
    real byte size. Payload = int16 clip + pick times + phase labels + header."""
    scale = float(np.max(np.abs(clip))) or 1.0
    clip_i16 = np.clip(clip / scale * 32767.0, -32768, 32767).astype(np.int16)
    header = json.dumps({
        "seg": seg, "t_s": round(float(t_s), 3), "gate": gate,
        "p_s": picks.get("P"), "s_s": picks.get("S"),
        "fs": float(fs), "scale": scale, "dtype": "int16",
    }).encode("utf-8")
    buf = io.BytesIO()
    np.savez_compressed(buf, clip=clip_i16, header=np.frombuffer(header, dtype=np.uint8))
    return len(buf.getvalue())


def fmt_bytes(n: float) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f} MB"
    if n >= 1_000:
        return f"{n/1_000:.1f} KB"
    return f"{int(n)} B"


@dataclass
class Counters:
    raw_bytes: float = 0.0
    raw_uncompressed_bytes: float = 0.0
    stalta_bytes: float = 0.0
    model_bytes: float = 0.0
    stalta_uncompressed_bytes: float = 0.0
    model_uncompressed_bytes: float = 0.0
    stalta_count: int = 0
    model_count: int = 0
    raw_count: int = 0
    stalta_false: int = 0
    model_false: int = 0
    incomplete_packets: int = 0
    elapsed_samples: int = 0


@dataclass
class Hop:
    seg: str
    t_s: float
    amp: float
    det: float
    ratio: float
    picks: dict
    clip: np.ndarray
    raw_chunk: np.ndarray
    in_event: object          # bool ground-truth, or None if unknown (live)
    seg_new: bool = False
    seg_total_s: float = 0.0
    seg_end: bool = False
    seg_summary: dict = None


@dataclass
class PendingTransmission:
    gate: str
    trigger_t_s: float
    due_t_s: float
    false: bool
    seg: str


class ContinuousWireCounter:
    """Encode continuous input in the same fixed-duration packets as gated data.

    This makes the bytes-on-wire comparison codec-for-codec instead of comparing
    compressed event packets with an uncompressed arithmetic baseline.
    """

    def __init__(self, clip_samples: int, fs: float):
        self.clip_samples = int(clip_samples)
        self.fs = float(fs)
        self.buffer = np.empty((3, 0), dtype=np.float32)
        self.bytes = 0
        self.packet_count = 0

    def push(self, chunk: np.ndarray, seg: str, t_s: float) -> None:
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[0] != 3:
            raise ValueError(f"continuous chunk must have shape (3,N), got {chunk.shape}")
        self.buffer = np.concatenate([self.buffer, chunk], axis=1)
        while self.buffer.shape[1] >= self.clip_samples:
            clip = self.buffer[:, :self.clip_samples]
            self.buffer = self.buffer[:, self.clip_samples:]
            self.bytes += serialize_packet(
                clip, {}, seg, t_s, fs=self.fs, gate="continuous"
            )
            self.packet_count += 1

    def finish_segment(self, seg: str, t_s: float) -> None:
        if self.buffer.shape[1]:
            self.bytes += serialize_packet(
                self.buffer, {}, seg, t_s, fs=self.fs, gate="continuous"
            )
            self.packet_count += 1
            self.buffer = np.empty((3, 0), dtype=np.float32)


# ---------------------------------------------------------------------------
# Sources -> streams of raw samples
# ---------------------------------------------------------------------------
def smoke_signal(cfg, seed: int = 42):
    """Deterministic synthetic accel stream: quiet MEMS baseline, one broadband
    'event' (both gates fire), and several non-seismic energy transients that
    STA/LTA over-triggers on but the model ignores (low-freq wobble / ramp bump).
    Returns (raw (3,N), event_windows [(s,s)]). ~90 s."""
    fs = float(cfg.data.sampling_rate)
    n = int(90 * fs)
    rng = np.random.default_rng(seed)
    raw = (0.05 * rng.standard_normal((3, n))).astype(np.float32)
    raw[0] += 1.0  # gravity offset on Z (demeaned away causally)

    # Real broadband "event": P at 45 s, S at 51 s -> both gates fire (TRUE).
    p_on, s_on = int(45 * fs), int(51 * fs)
    for ch in range(3):
        raw[ch] += arrival_wave(n, fs, p_on, 9.0, 6.0, 1.0)
        raw[ch] += arrival_wave(n, fs, s_on, 3.0, 8.0, 3.0)
    event_windows = [(43.0, 58.0)]

    # Non-seismic transients OUTSIDE the event window: sub-1 Hz "wobbles" (cart
    # rolls by, door slam settle, thermal tilt). STA/LTA has no frequency
    # selectivity -> its broadband energy ratio fires; the model's 1-45 Hz
    # passband attenuates sub-1 Hz -> it stays quiet. This is the honest smoke
    # mechanism (verified: model detmax < 0.80, STA/LTA ratio > 3.0).
    def wobble(t0, dur, freq, amp):
        a, b = int(t0 * fs), int((t0 + dur) * fs)
        env = np.hanning(b - a)
        raw[:, a:b] += amp * env * np.sin(2 * np.pi * freq * np.arange(b - a) / fs)

    wobble(12.0, 2.5, 0.6, 1.2)
    wobble(25.0, 2.5, 0.5, 1.2)
    wobble(68.0, 2.5, 0.7, 1.2)
    wobble(80.0, 2.5, 0.6, 1.2)
    return raw.astype(np.float32), event_windows


def _s16(high: int, low: int) -> int:
    value = (high << 8) | low
    return value - 65536 if value >= 32768 else value


class TimedI2CSensor:
    """Small common interface for blocking, paced I2C sample acquisition."""

    def read_sample(self) -> tuple[float, float, float]:
        raise NotImplementedError

    def read_block(self, n: int, fs: float) -> np.ndarray:
        out = np.empty((3, n), dtype=np.float32)
        dt = 1.0 / fs
        deadline = time.perf_counter()
        for i in range(n):
            out[:, i] = self.read_sample()
            deadline += dt
            rem = deadline - time.perf_counter()
            if rem > 0:
                time.sleep(rem)
        return out


class MPU6050(TimedI2CSensor):
    """Minimal I2C MPU6050 reader (smbus2). +/-2 g (AFS_SEL=0), on-chip 100 Hz
    (SMPLRT_DIV=9 + DLPF). Axis convention (NOT calibration): az->Z, ax->N, ay->E."""

    ADDR = 0x68
    PWR_MGMT_1 = 0x6B
    SMPLRT_DIV = 0x19
    CONFIG = 0x1A
    ACCEL_CONFIG = 0x1C
    ACCEL_XOUT_H = 0x3B

    def __init__(self, bus_id: int = 1, address: int = ADDR):
        from smbus2 import SMBus  # raises if unavailable -> caller handles it
        self.bus = SMBus(bus_id)
        self.address = int(address)
        self.bus.write_byte_data(self.address, self.PWR_MGMT_1, 0x00)   # wake
        self.bus.write_byte_data(self.address, self.CONFIG, 0x03)       # DLPF ~44 Hz
        self.bus.write_byte_data(self.address, self.SMPLRT_DIV, 0x09)   # 1kHz/(1+9)=100Hz
        self.bus.write_byte_data(self.address, self.ACCEL_CONFIG, 0x00)  # +/-2 g

    def read_sample(self):
        d = self.bus.read_i2c_block_data(self.address, self.ACCEL_XOUT_H, 6)
        ax = _s16(d[0], d[1]) / 16384.0
        ay = _s16(d[2], d[3]) / 16384.0
        az = _s16(d[4], d[5]) / 16384.0
        return az, ax, ay   # Z, N, E convention


class MPU3050(TimedI2CSensor):
    """Minimal MPU3050 gyroscope reader.

    Output is X/Y/Z angular rate in degrees/second. These channels are deliberately
    not relabeled Z/N/E: a gyroscope is a live motion-pipeline demonstration, not
    a substitute for the translational seismic input used to train the model.
    Registers and the 131 LSB/(degree/s) scale follow the InvenSense MPU-3050
    product specification and register map (FS_SEL=0, +/-250 degree/s).
    """

    ADDR = 0x68
    WHO_AM_I = 0x00
    SMPLRT_DIV = 0x15
    DLPF_FS_SYNC = 0x16
    GYRO_XOUT_H = 0x1D
    PWR_MGM = 0x3E
    SENSITIVITY_LSB_PER_DPS = 131.0

    def __init__(self, bus_id: int = 1, address: int = ADDR):
        from smbus2 import SMBus  # raises if unavailable -> caller handles it
        self.bus = SMBus(bus_id)
        self.address = int(address)
        self.bus.write_byte_data(self.address, self.PWR_MGM, 0x00)
        time.sleep(0.05)
        # FS_SEL=0 (+/-250 dps), DLPF_CFG=3 (~42 Hz, 1 kHz internal sample rate).
        self.bus.write_byte_data(self.address, self.DLPF_FS_SYNC, 0x03)
        self.bus.write_byte_data(self.address, self.SMPLRT_DIV, 0x09)
        self.who_am_i = self.bus.read_byte_data(self.address, self.WHO_AM_I)

    def read_sample(self):
        d = self.bus.read_i2c_block_data(self.address, self.GYRO_XOUT_H, 6)
        scale = self.SENSITIVITY_LSB_PER_DPS
        return (
            _s16(d[0], d[1]) / scale,
            _s16(d[2], d[3]) / scale,
            _s16(d[4], d[5]) / scale,
        )


# ---------------------------------------------------------------------------
# Hop generators (all sources -> Hop stream consumed by one loop)
# ---------------------------------------------------------------------------
def _predictor(session, name_in, name_out):
    def predict(x):
        return session.run([name_out], {name_in: x.astype(np.float32)})[0]
    return predict


def _stalta(raw, fs):
    """STA/LTA on demeaned data. Classic detectors run on ~zero-mean velocity;
    demeaning removes the accelerometer's gravity DC (which would otherwise flood
    the energy ratio) so the detector responds to bursts, not the static tilt."""
    demeaned = raw - raw.mean(axis=1, keepdims=True)
    return sta_lta_ratio(demeaned, fs, STA_S, LTA_S)


def _window_picks(probs, fs, seg_t0_s):
    """Best P/S pick times (absolute stream seconds) in the current window."""
    picks = extract_phase_picks(probs, fs, P_THRESHOLD, S_THRESHOLD, 1.0)
    out = {"P": None, "S": None}
    for phase in ("P", "S"):
        cands = [p for p in picks if p["phase"] == phase]
        if cands:
            best = max(cands, key=lambda p: p["probability"])
            out[phase] = round(seg_t0_s + best["time_s"], 2)
    return out


def _tile_tail(x: np.ndarray, length: int) -> np.ndarray:
    if x.shape[1] < 1:
        raise ValueError("cannot warm-start a rolling buffer from no samples")
    return np.tile(x, (1, length // x.shape[1] + 1))[:, -length:].astype(np.float32)


def _online_hops(
    cfg,
    chunks,
    ambient,
    total_samples,
    event_windows,
    predict,
    hop,
    window,
    seg="LIVE",
):
    """Process newly acquired chunks once with persistent causal state.

    ``chunks`` stays lazy, so a physical sensor can acquire one hop, infer it, and
    only then acquire the next. Both processed and raw rolling buffers are seeded
    from measured ambient samples rather than zeros.
    """
    fs = float(cfg.data.sampling_rate)
    ambient = np.asarray(ambient, dtype=np.float32)
    pre = CausalPreprocessor(
        cfg,
        warmup_samples=min(int(fs), ambient.shape[1]),
        initial_background=ambient,
    )
    processed_buf = _tile_tail(pre.process(ambient), window)
    raw_model_buf = _tile_tail(ambient, window)
    clip_n = int(round((CLIP_PRE_S + CLIP_POST_S) * fs))
    # Packet history is never padded: an early trigger is dropped unless three
    # seconds of genuinely acquired pre-trigger data exists.
    raw_clip_buf = ambient[:, -clip_n:].copy()
    total_s = total_samples / fs
    pos = ambient.shape[1]
    first = True
    for chunk in chunks:
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[0] != 3:
            raise ValueError(f"source chunk must have shape (3,N), got {chunk.shape}")
        h = chunk.shape[1]
        if h == 0:
            continue
        processed = pre.process(chunk)
        processed_buf = np.roll(processed_buf, -h, axis=1)
        processed_buf[:, -h:] = processed
        raw_model_buf = np.roll(raw_model_buf, -h, axis=1)
        raw_model_buf[:, -h:] = chunk
        raw_clip_buf = np.concatenate([raw_clip_buf, chunk], axis=1)[:, -clip_n:]
        pos += h
        t_s = pos / fs
        probs = predict(processed_buf[None, ...])[0]
        det = float(probs[0, -h:].max())
        ratio = float(_stalta(raw_model_buf, fs)[-h:].max())
        picks = {}   # live/synthetic: detection only; real P/S is shown via --source replay
        in_event = None
        if event_windows is not None:
            in_event = any(a <= t_s <= b for a, b in event_windows)
        accounting_chunk = np.concatenate([ambient, chunk], axis=1) if first else chunk
        yield Hop(
            seg=seg,
            t_s=t_s,
            amp=float(np.abs(chunk).max()),
            det=det,
            ratio=ratio,
            picks=picks,
            clip=raw_clip_buf.copy(),
            raw_chunk=accounting_chunk,
            in_event=in_event,
            seg_new=first,
            seg_total_s=total_s,
            seg_end=pos >= total_samples,
        )
        first = False


def live_hops(cfg, raw_full, event_windows, predict, warmup_s, hop, window):
    """Array-backed version of the persistent online path (smoke/tests)."""
    raw_full = np.asarray(raw_full, dtype=np.float32)
    n = raw_full.shape[1]
    warmup_n = min(max(1, int(warmup_s)), n)
    ambient = raw_full[:, :warmup_n]

    def chunks():
        for pos in range(warmup_n, n, hop):
            yield raw_full[:, pos:min(pos + hop, n)]

    return _online_hops(
        cfg, chunks(), ambient, n, event_windows, predict, hop, window
    )


def device_hops(cfg, dev, args, predict, warmup_n, hop, window):
    """Interleave physical acquisition and inference one hop at a time."""
    fs = float(cfg.data.sampling_rate)
    capture_n = int(round(args.duration_s * fs))
    print(
        f"[{args.source}] live acquisition {args.duration_s:.0f}s @ {fs:.0f}Hz "
        f"after {args.warmup_s:.0f}s measured ambient warm-up"
    )
    ambient = dev.read_block(warmup_n, fs)

    def chunks():
        remaining = capture_n
        while remaining > 0:
            n = min(hop, remaining)
            yield dev.read_block(n, fs)
            remaining -= n

    return _online_hops(
        cfg,
        chunks(),
        ambient,
        warmup_n + capture_n,
        None,
        predict,
        hop,
        window,
        seg=args.source.upper(),
    )


def replay_hops(cfg, bundle, predict, hop):
    """Replay path: each REAL STEAD trace scored with causal_stream_probabilities
    (the validated streaming-equivalent path), then revealed left->right in hops."""
    fs = float(cfg.data.sampling_rate)
    raws = bundle["raw"]
    ids = bundle["trace_id"]
    is_eq = bundle["is_eq"]
    p_samp = bundle["p_sample"]
    s_samp = bundle["s_sample"]
    n_trace = raws.shape[0]
    for ti in range(n_trace):
        raw = raws[ti].astype(np.float32)
        n = raw.shape[1]
        pre = CausalPreprocessor(cfg, warmup_samples=min(int(fs), hop))
        probs = causal_stream_probabilities(raw, predict, pre, chunk_samples=500).probabilities
        ratio = _stalta(raw, fs)
        seg = f"REPLAY {ti+1}/{n_trace} {ids[ti]}"
        if bool(is_eq[ti]):
            ew = [((p_samp[ti]) / fs - 1.0, (s_samp[ti]) / fs + 3.0)]
        else:
            ew = []   # pure noise trace: any trigger is a false transmission
        seg_picks = _window_picks(probs, fs, 0.0)
        summary = None
        if bool(is_eq[ti]):
            summary = {
                "model_p": seg_picks["P"], "model_s": seg_picks["S"],
                "catalog_p": round(float(p_samp[ti]) / fs, 2),
                "catalog_s": round(float(s_samp[ti]) / fs, 2),
            }
        starts = list(range(0, n, hop))
        for pos in starts:
            end = min(pos + hop, n)
            t_s = end / fs
            det = float(probs[0, pos:end].max())
            rat = float(ratio[pos:end].max())
            in_event = any(a <= t_s <= b for a, b in ew)
            a = max(0, end - int((CLIP_PRE_S + CLIP_POST_S) * fs))
            clip = raw[:, a:end]
            # only surface a pick once its sample has been reached (real-time causal)
            picks = {"P": seg_picks["P"] if (seg_picks["P"] is not None and seg_picks["P"] <= t_s) else None,
                     "S": seg_picks["S"] if (seg_picks["S"] is not None and seg_picks["S"] <= t_s) else None}
            is_last = pos == starts[-1]
            yield Hop(
                seg=seg,
                t_s=t_s,
                amp=float(np.abs(raw[:, pos:end]).max()),
                det=det,
                ratio=rat,
                picks=picks,
                clip=clip,
                raw_chunk=raw[:, pos:end],
                in_event=in_event,
                seg_new=(pos == 0),
                seg_total_s=n / fs,
                seg_end=is_last,
                seg_summary=summary if is_last else None,
            )


# ---------------------------------------------------------------------------
# Consumer: gating, byte accounting, rendering
# ---------------------------------------------------------------------------
def run(hops, cfg, args, source_label, plotter=None):
    fs = float(cfg.data.sampling_rate)
    c = Counters()
    model_trig = EdgeTrigger(DET_THRESHOLD, off=0.6, refractory_s=REFRACTORY_S)
    stalta_trig = EdgeTrigger(STA_LTA_ON, off=STA_LTA_ON, refractory_s=REFRACTORY_S)
    clip_samples = int(round((CLIP_PRE_S + CLIP_POST_S) * fs))
    wire = ContinuousWireCounter(clip_samples, fs)
    pending: list[PendingTransmission] = []
    last_status_t = -1e9
    transmissions = []

    print(f"\n=== EDGE TRANSMISSION-GATING DEMO  [source: {source_label}] ===")
    print(f"gates: MODEL det>={DET_THRESHOLD:.2f}   STA/LTA ratio>={STA_LTA_ON:.1f} "
          f"(sta={STA_S}s lta={LTA_S}s)   packet=3s+7s clip, {BYTES_PER_SAMPLE}B/sample")
    print("wire comparison = identical int16 + compressed-NPZ encoding for continuous "
          "and gated paths\n")

    for h in hops:
        if h.seg_new:
            if pending:
                c.incomplete_packets += len(pending)
                pending.clear()
            model_trig.reset()
            stalta_trig.reset()
            last_status_t = -1e9
            print(f"\n--- {h.seg}  (segment length {h.seg_total_s:.0f}s) ---")

        new_samples = int(h.raw_chunk.shape[1])
        c.elapsed_samples += new_samples
        c.raw_uncompressed_bytes += new_samples * 3 * BYTES_PER_SAMPLE
        wire.push(h.raw_chunk, h.seg, h.t_s)
        c.raw_bytes = wire.bytes
        c.raw_count = wire.packet_count

        m_fire = model_trig.update(h.det, h.t_s)
        s_fire = stalta_trig.update(h.ratio, h.t_s)

        if s_fire:
            false = (h.in_event is False)
            tag = "  [FALSE ALARM]" if false else ("  [event]" if h.in_event else "")
            print(f"  \U0001F514 STA/LTA trigger  @{h.t_s:6.1f}s; collecting {CLIP_POST_S:g}s post-trigger{tag}")
            pending.append(PendingTransmission(
                "stalta", h.t_s, h.t_s + CLIP_POST_S, false, h.seg
            ))
        if m_fire:
            false = (h.in_event is False)
            tag = "  [FALSE ALARM]" if false else ("  [event]" if h.in_event else "")
            print(f"  \U0001F514 MODEL trigger    @{h.t_s:6.1f}s; collecting {CLIP_POST_S:g}s post-trigger{tag}")
            pending.append(PendingTransmission(
                "model", h.t_s, h.t_s + CLIP_POST_S, false, h.seg
            ))

        ready = [p for p in pending if p.seg == h.seg and h.t_s >= p.due_t_s]
        for p in ready:
            if h.clip.shape[1] != clip_samples:
                c.incomplete_packets += 1
                pending.remove(p)
                continue
            pkt = serialize_packet(
                h.clip, h.picks, h.seg, p.trigger_t_s, fs=fs, gate=p.gate
            )
            payload = int(h.clip.size * BYTES_PER_SAMPLE)
            if p.gate == "stalta":
                c.stalta_bytes += pkt
                c.stalta_uncompressed_bytes += payload
                c.stalta_count += 1
                c.stalta_false += int(p.false)
            else:
                c.model_bytes += pkt
                c.model_uncompressed_bytes += payload
                c.model_count += 1
                c.model_false += int(p.false)
            tag = "  [FALSE ALARM]" if p.false else ""
            print(
                f"  \U0001F4E1 {p.gate.upper():7s} transmit @{h.t_s:6.1f}s "
                f"(trigger {p.trigger_t_s:5.1f}s) packet={fmt_bytes(pkt):>7s}{tag}"
            )
            transmissions.append({
                "gate": p.gate,
                "trigger_t_s": round(p.trigger_t_s, 2),
                "transmit_t_s": round(h.t_s, 2),
                "bytes": int(pkt),
                "uncompressed_payload_bytes": payload,
                "clip_samples": int(h.clip.shape[1]),
                "false": bool(p.false),
            })
            pending.remove(p)

        if h.t_s - last_status_t >= 1.0:
            last_status_t = h.t_s
            sl_arm = "ARMED" if stalta_trig.armed else "  .  "
            md_arm = "ARMED" if model_trig.armed else "  .  "
            print(f"[t={h.t_s:6.1f}s] amp={h.amp:5.2f}  "
                  f"STA/LTA {h.ratio:5.1f} {sl_arm}  MODEL det={h.det:.2f} {md_arm}  |  "
                  f"TX raw {fmt_bytes(c.raw_bytes):>8s}  "
                  f"stalta {fmt_bytes(c.stalta_bytes):>7s}  "
                  f"model {fmt_bytes(c.model_bytes):>7s}")

        if h.seg_end and h.seg_summary:
            s = h.seg_summary
            def fp(v):
                return "--" if v is None else f"{v:.1f}s"
            print(f"     ✓ model recovered: P {fp(s['model_p'])}  S {fp(s['model_s'])}"
                  f"   (catalog P {fp(s['catalog_p'])}  S {fp(s['catalog_s'])})  [REAL STEAD]")

        if h.seg_end:
            wire.finish_segment(h.seg, h.t_s)
            c.raw_bytes = wire.bytes
            c.raw_count = wire.packet_count
            incomplete = [p for p in pending if p.seg == h.seg]
            if incomplete:
                c.incomplete_packets += len(incomplete)
                pending = [p for p in pending if p.seg != h.seg]
                print(f"  [segment ended before {len(incomplete)} pending packet(s) collected full post-trigger data]")

        if plotter is not None:
            plotter.update(h, c, stalta_trig.armed, model_trig.armed)

        if args.source in ("smoke", "replay") and args.speed > 0 and not args.fast:
            time.sleep((new_samples / fs) / args.speed)

    report = build_report(c, source_label, transmissions, fs)
    return c, report


def build_report(c: Counters, source_label: str, transmissions, fs: float = 100.0):
    def ratio(a, b):
        return None if b <= 0 else round(a / b, 1)
    return {
        "source": source_label,
        "bytes_per_sample": BYTES_PER_SAMPLE,
        "packet_clip_s": CLIP_PRE_S + CLIP_POST_S,
        "gates": {"model_detection_threshold": DET_THRESHOLD,
                  "stalta": {"sta_s": STA_S, "lta_s": LTA_S, "on": STA_LTA_ON}},
        "elapsed_seconds": round(c.elapsed_samples / fs, 1),
        "encoding": "float input scaled to int16 per packet, then np.savez_compressed; same codec for all paths",
        "bytes": {
            "raw_continuous": int(c.raw_bytes),
            "stalta_gated": int(c.stalta_bytes),
            "model_gated": int(c.model_bytes),
        },
        "uncompressed_payload_bytes": {
            "raw_continuous": int(c.raw_uncompressed_bytes),
            "stalta_gated": int(c.stalta_uncompressed_bytes),
            "model_gated": int(c.model_uncompressed_bytes),
        },
        "transmissions": {
            "raw_continuous": {"count": c.raw_count},
            "stalta": {"count": c.stalta_count, "false": c.stalta_false},
            "model": {"count": c.model_count, "false": c.model_false},
            "incomplete_dropped": c.incomplete_packets,
        },
        "reduction_vs_raw": {
            "stalta": ratio(c.raw_bytes, c.stalta_bytes),
            "model": ratio(c.raw_bytes, c.model_bytes),
        },
        "model_vs_stalta_bytes_x": ratio(c.stalta_bytes, c.model_bytes),
        "transmission_log": transmissions,
    }


# ---------------------------------------------------------------------------
# Stage 6: headline three-bar chart (status palette: raw=critical, stalta=warning,
# model=good). Value + reduction labels satisfy the low-contrast relief rule.
# ---------------------------------------------------------------------------
def make_bytes_chart(report, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
    CRITICAL, WARNING, GOOD = "#d03b3b", "#eda100", "#0ca30c"
    b = report["bytes"]
    vals = [b["raw_continuous"], b["stalta_gated"], b["model_gated"]]
    labels = ["Raw continuous\n(transmit everything)",
              "STA/LTA gated\n(energy detector)",
              "Model gated\n(causal INT8 U-Net)"]
    colors = [CRITICAL, WARNING, GOOD]

    fig, ax = plt.subplots(figsize=(10, 5.0))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    y = np.arange(3)[::-1]
    bars = ax.barh(y, vals, color=colors, height=0.62, zorder=3)
    for rect in bars:
        rect.set_linewidth(0)

    r = report["reduction_vs_raw"]
    tx = report["transmissions"]
    ann = [
        "baseline",
        (f"{r['stalta']}x less than raw" if r["stalta"] else "")
        + f"   • {tx['stalta']['count']} transmits"
        + (f" ({tx['stalta']['false']} false)" if tx['stalta']['false'] else ""),
        (f"{r['model']}x less than raw" if r["model"] else "")
        + f"   • {tx['model']['count']} transmits"
        + (f" ({tx['model']['false']} false)" if tx['model']['false'] else ""),
    ]
    xmax = max(vals) if max(vals) > 0 else 1
    for yi, v, a in zip(y, vals, ann):
        ax.text(v + xmax * 0.012, yi, f"  {fmt_bytes(v)}", va="center", ha="left",
                fontsize=12, fontweight="bold", color=INK, zorder=4)
        ax.text(v + xmax * 0.012, yi - 0.28, f"  {a}", va="center", ha="left",
                fontsize=9, color=MUTED, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.5, color=INK)
    ax.set_xlim(0, xmax * 1.22)
    ax.set_xlabel("Encoded wire bytes (same int16 + compressed-NPZ codec for every path)",
                  fontsize=10, color=MUTED)
    mvs = report.get("model_vs_stalta_bytes_x")
    display_source = (
        "real STEAD replay" if report["source"].startswith("replay")
        else report["source"]
    )
    sub = f"{display_source}  •  {report['elapsed_seconds']:.0f}s"
    if mvs:
        sub += f"  •  model sends {mvs}x fewer bytes than STA/LTA"
    ax.set_title("Bytes on the wire: transmission gating\n" + sub,
                 fontsize=13, fontweight="bold", color=INK, loc="left")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d8d7d2")
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color="#ecebe7", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Optional live plot (dependency-light): scrolling amplitude + two alarm lanes.
# ---------------------------------------------------------------------------
class Plotter:
    def __init__(self, save_frame: Path | None, window_s=30.0, fs=100.0):
        import matplotlib
        self.interactive = save_frame is None
        if not self.interactive:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        self.save_frame = save_frame
        self.window_s = window_s
        self.fs = float(fs)
        self.T, self.A, self.D, self.R = [], [], [], []
        self.tx_model, self.tx_stalta = [], []
        self.fig, self.axes = plt.subplots(
            3, 1, figsize=(10, 6), sharex=True,
            gridspec_kw={"height_ratios": [3, 1, 1]})
        self.fig.patch.set_facecolor("#fcfcfb")
        if self.interactive:
            plt.ion()
            self.fig.show()
        self._draw_count = 0

    def update(self, h, c, sl_armed, md_armed):
        self.T.append(c.elapsed_samples / self.fs)   # monotonic session clock
        self.A.append(abs(h.amp))
        self.D.append(h.det)
        self.R.append(min(h.ratio, 10.0))
        self._redraw(c)

    def _redraw(self, c):
        self._draw_count += 1
        if self.interactive and self._draw_count % 2:
            return
        ax_w, ax_s, ax_m = self.axes
        for ax in self.axes:
            ax.clear()
            ax.set_facecolor("#fcfcfb")
        t = np.array(self.T)
        if self.interactive:
            lo = max(0.0, (t[-1] if len(t) else 0) - self.window_s)
            m = t >= lo
        else:
            m = np.ones(len(t), dtype=bool)   # save mode: show the whole session
        A = np.array(self.A)
        A = A / (A.max() or 1.0)   # normalize (STEAD counts vs g are per-source)
        ax_w.plot(t[m], A[m], color="#2a78d6", lw=1.2)
        ax_w.set_ylabel("|signal|\n(norm.)")
        ax_w.set_title("edge transmission gating  —  scrolling waveform + alarm lanes",
                       loc="left", fontsize=11, fontweight="bold")
        ax_s.plot(t[m], np.array(self.R)[m], color="#eda100", lw=1.5)
        ax_s.axhline(STA_LTA_ON, color="#eda100", ls="--", lw=0.8, alpha=0.6)
        ax_s.set_ylabel("STA/LTA")
        ax_s.set_ylim(0, 10.5)
        ax_m.plot(t[m], np.array(self.D)[m], color="#0ca30c", lw=1.5)
        ax_m.axhline(DET_THRESHOLD, color="#0ca30c", ls="--", lw=0.8, alpha=0.6)
        ax_m.set_ylabel("model det")
        ax_m.set_ylim(0, 1.05)
        ax_m.set_xlabel("time (s)")
        txt = (f"raw {fmt_bytes(c.raw_bytes)}   "
               f"STA/LTA {fmt_bytes(c.stalta_bytes)} ({c.stalta_count} tx)   "
               f"model {fmt_bytes(c.model_bytes)} ({c.model_count} tx)")
        self.fig.suptitle(txt, y=0.02, fontsize=10, color="#52514e")
        if self.interactive:
            self.plt.pause(0.001)

    def finalize(self):
        if not self.interactive and self.save_frame is not None:
            self.fig.tight_layout(rect=(0, 0.04, 1, 1))
            self.fig.savefig(self.save_frame, dpi=130, facecolor="#fcfcfb")
            print(f"saved plot preview -> {self.save_frame}")
        self.plt.close(self.fig)


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Live MPU3050/MPU6050 or replay edge demo.")
    p.add_argument(
        "--source", choices=["mpu3050", "mpu6050", "smoke", "replay"],
        default="smoke",
    )
    p.add_argument("--trace", default=str(DEMO_DIR / "replay_traces.npz"),
                   help="replay bundle (default: outputs/demo/replay_traces.npz)")
    p.add_argument("--onnx", default=str(DEMO_DIR / "causal_stage3_int8.onnx"))
    p.add_argument("--config", default=str(REPO / "configs/default.yaml"))
    p.add_argument("--speed", type=float, default=4.0,
                   help="playback speed multiple (smoke/replay pacing); higher=faster")
    p.add_argument("--fast", action="store_true", help="no pacing (quick verification)")
    p.add_argument("--warmup-s", type=float, default=3.0,
                   help="seconds of real ambient collected to warm-start (live path)")
    p.add_argument("--plot", action="store_true", help="live scrolling plot (needs display)")
    p.add_argument("--save-frame", default=None,
                   help="render the plot view to a PNG instead of a live window (headless)")
    p.add_argument("--report", default=str(DEMO_DIR / "bytes_report.json"))
    p.add_argument("--chart", default=str(DEMO_DIR / "bytes_comparison.png"))
    p.add_argument("--duration-s", type=float, default=90.0,
                   help="live hardware acquisition length after ambient warm-up")
    p.add_argument("--i2c-bus", type=int, default=1)
    p.add_argument("--i2c-address", type=lambda value: int(value, 0), default=0x68,
                   help="sensor I2C address, decimal or 0x-prefixed (default: 0x68)")
    p.add_argument(
        "--fallback-smoke", action="store_true",
        help="if a requested hardware source is unavailable, explicitly fall back to smoke",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    fs = float(cfg.data.sampling_rate)
    window = int(cfg.data.window_samples)
    hop = int(round(HOP_S * fs))
    warmup_n = int(round(args.warmup_s * fs))
    if warmup_n < 1:
        raise SystemExit("--warmup-s must provide at least one measured sample")

    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        raise SystemExit(f"causal INT8 ONNX not found: {onnx_path} "
                         "(run export_onnx.py + quantize_onnx.py --causal)")
    session, name_in, name_out = make_session(onnx_path, 1)
    predict = _predictor(session, name_in, name_out)

    # ---- resolve source -> hop generator ----------------------------------
    source_label = args.source
    if args.source == "replay":
        bundle = np.load(args.trace, allow_pickle=True)
        source_label = f"replay ({args.trace.split('/')[-1]}, REAL STEAD)"
        hops = replay_hops(cfg, bundle, predict, hop)
    elif args.source in ("mpu3050", "mpu6050"):
        try:
            if args.source == "mpu3050":
                dev = MPU3050(args.i2c_bus, args.i2c_address)
                source_label = (
                    f"mpu3050 (live I2C 0x{args.i2c_address:02x}, XYZ angular rate dps; "
                    "motion demo, not seismic validation)"
                )
            else:
                dev = MPU6050(args.i2c_bus, args.i2c_address)
                source_label = "mpu6050 (live I2C, +/-2g acceleration; motion demo)"
            hops = device_hops(cfg, dev, args, predict, warmup_n, hop, window)
        except Exception as exc:
            message = f"{args.source} unavailable: {exc.__class__.__name__}: {exc}"
            if not args.fallback_smoke:
                raise SystemExit(message + " (use --fallback-smoke to opt into synthetic fallback)")
            print(f"[{message}] -> explicit smoke fallback")
            source_label = f"smoke ({args.source} fallback)"
            raw_full, event_windows = smoke_signal(cfg, args.seed)
            hops = live_hops(
                cfg, raw_full, event_windows, predict, warmup_n, hop, window
            )
    else:
        raw_full, event_windows = smoke_signal(cfg, args.seed)
        hops = live_hops(cfg, raw_full, event_windows, predict, warmup_n, hop, window)

    plotter = None
    if args.plot or args.save_frame:
        import os
        save_to = args.save_frame
        if args.plot and not save_to and not os.environ.get("DISPLAY"):
            save_to = str(DEMO_DIR / "plot_preview.png")
            print(f"[--plot: no DISPLAY -> rendering a static preview to {save_to}]")
        DEMO_DIR.mkdir(parents=True, exist_ok=True)
        plotter = Plotter(Path(save_to) if save_to else None, fs=fs)

    c, report = run(hops, cfg, args, source_label, plotter=plotter)

    # ---- Stage 6: outputs --------------------------------------------------
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))
    make_bytes_chart(report, Path(args.chart))
    if plotter is not None:
        plotter.finalize()

    print("\n=== SESSION SUMMARY ===")
    print(f"  elapsed          {report['elapsed_seconds']:.0f}s")
    print(f"  raw continuous   {fmt_bytes(c.raw_bytes)}")
    print(f"  STA/LTA gated    {fmt_bytes(c.stalta_bytes)}   "
          f"({c.stalta_count} transmits, {c.stalta_false} false)")
    print(f"  model gated      {fmt_bytes(c.model_bytes)}   "
          f"({c.model_count} transmits, {c.model_false} false)")
    if report["model_vs_stalta_bytes_x"]:
        print(f"  model sends {report['model_vs_stalta_bytes_x']}x fewer bytes than STA/LTA")
    print(f"\nwrote {args.report}\n      {args.chart}")


if __name__ == "__main__":
    main()
