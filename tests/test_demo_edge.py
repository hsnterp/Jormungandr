"""Regression tests for the edge demo's hardware and wire-accounting contracts."""

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import demo_edge as D  # noqa: E402
from seismic_edge_picker.config import load_config  # noqa: E402


def _cfg():
    return load_config(str(Path(__file__).resolve().parents[1] / "configs/default.yaml"))


def test_mpu3050_uses_gyro_register_map_and_scale(monkeypatch):
    class FakeBus:
        def __init__(self, bus_id):
            self.bus_id = bus_id
            self.writes = []

        def write_byte_data(self, address, register, value):
            self.writes.append((address, register, value))

        def read_byte_data(self, address, register):
            return 0x68

        def read_i2c_block_data(self, address, register, length):
            assert (address, register, length) == (0x68, 0x1D, 6)
            # +131, -131, +262 -> +1, -1, +2 degree/s.
            return [0x00, 0x83, 0xFF, 0x7D, 0x01, 0x06]

    bus = FakeBus(1)
    monkeypatch.setitem(sys.modules, "smbus2", SimpleNamespace(SMBus=lambda _: bus))
    monkeypatch.setattr(D.time, "sleep", lambda _: None)

    sensor = D.MPU3050(bus_id=1, address=0x68)
    assert bus.writes == [
        (0x68, 0x3E, 0x00),
        (0x68, 0x16, 0x03),
        (0x68, 0x15, 0x09),
    ]
    assert sensor.who_am_i == 0x68
    assert np.allclose(sensor.read_sample(), (1.0, -1.0, 2.0))


def test_live_hops_processes_each_sample_once_with_persistent_state(monkeypatch):
    cfg = _cfg()
    raw = np.arange(3 * 400, dtype=np.float32).reshape(3, 400)
    calls = []

    class FakePreprocessor:
        def __init__(self, *args, **kwargs):
            pass

        def process(self, chunk):
            calls.append(chunk.shape[1])
            return np.asarray(chunk, dtype=np.float32)

    monkeypatch.setattr(D, "CausalPreprocessor", FakePreprocessor)

    def predict(batch):
        return np.zeros_like(batch, dtype=np.float32)

    hops = list(D.live_hops(
        cfg, raw, None, predict, warmup_s=100, hop=50, window=200
    ))
    assert calls == [100] + [50] * 6
    assert sum(h.raw_chunk.shape[1] for h in hops) == raw.shape[1]
    assert hops[0].seg_new is True
    assert hops[0].clip.shape[1] == 150  # measured warm-up + first chunk, no padding
    assert hops[-1].seg_end is True


def test_packet_waits_for_full_post_trigger_and_has_fixed_clip():
    cfg = _cfg()
    fs = float(cfg.data.sampling_rate)
    hop_n = int(D.HOP_S * fs)
    clip_n = int((D.CLIP_PRE_S + D.CLIP_POST_S) * fs)
    raw = np.linspace(-1, 1, 3 * 1200, dtype=np.float32).reshape(3, 1200)

    hops = []
    for pos in range(0, raw.shape[1], hop_n):
        end = min(pos + hop_n, raw.shape[1])
        t_s = end / fs
        hops.append(D.Hop(
            seg="TEST",
            t_s=t_s,
            amp=1.0,
            det=0.95 if t_s == 4.0 else 0.0,
            ratio=0.0,
            picks={},
            clip=raw[:, max(0, end - clip_n):end],
            raw_chunk=raw[:, pos:end],
            in_event=True,
            seg_new=(pos == 0),
            seg_total_s=12.0,
            seg_end=(end == raw.shape[1]),
        ))

    args = SimpleNamespace(source="smoke", speed=0.0, fast=True)
    counters, report = D.run(iter(hops), cfg, args, "test")

    assert counters.model_count == 1
    assert counters.incomplete_packets == 0
    tx = report["transmission_log"][0]
    assert tx["gate"] == "model"
    assert tx["trigger_t_s"] == 4.0
    assert tx["transmit_t_s"] == 11.0
    assert tx["clip_samples"] == clip_n
    assert tx["uncompressed_payload_bytes"] == 3 * clip_n * D.BYTES_PER_SAMPLE
    assert report["bytes"]["raw_continuous"] > report["bytes"]["model_gated"]


def test_continuous_wire_counter_uses_same_ten_second_packetization():
    fs = 100.0
    clip_n = 1000
    counter = D.ContinuousWireCounter(clip_n, fs)
    raw = np.ones((3, 2000), dtype=np.float32)
    counter.push(raw[:, :750], "TEST", 7.5)
    counter.push(raw[:, 750:], "TEST", 20.0)
    counter.finish_segment("TEST", 20.0)
    assert counter.packet_count == 2
    assert counter.buffer.shape == (3, 0)
    assert counter.bytes > 0
