# Live edge demo — MPU6050 transmission gating

`scripts/demo_edge.py` runs the project's causal INT8 phase-picker as an on-device
**transmission gate** for a bandwidth-limited remote sensor, and compares it head
to head with a classic STA/LTA energy detector. The headline is **bytes on the
wire**: STA/LTA false-triggers on any energy burst and transmits far more; the
learned model is selective and transmits far less for the same real events.

Three source modes feed one identical pipeline:

| `--source` | what it is | shows |
|---|---|---|
| `smoke`   | deterministic synthetic accel stream (no hardware, no dataset) | both gates behave; byte counters diverge |
| `mpu6050` | live I²C read from a physical MPU6050 (Raspberry Pi) | the full pipeline runs in real time on edge hardware and a physical vibration triggers it |
| `replay`  | **real STEAD earthquakes** streamed through the same causal path | genuine P **and** S phase picks the accelerometer cannot produce |

---

## 🎥 Recorded demo — the browser view (recommended for presenting)

The console + PNG output is fine for a terminal, but for a **recording** use the
self-contained web view. It replays a real `--source replay` run and animates it
live in a clean, white page: each real STEAD earthquake streams left→right with
the **P/S picks landing** as the waves arrive, the **model's detection confidence**
rising through the event, the **STA/LTA lane** underneath, and the **bytes-on-the-
wire bars growing** as each gate fires — ending on the noise trace where the model
stays silent.

```bash
# one time (re-scores the real STEAD replay bundle through the causal pipeline):
python3 scripts/export_gating_traces.py
# then just open it — no server, no dependencies, works over file://:
open outputs/demo/gating_demo.html
```

`export_gating_traces.py` reuses `demo_edge.py`'s pipeline verbatim, so the byte
numbers match a `--source replay` run. It writes two files into `outputs/demo/`:
`gating_traces.json` (the data) and `gating_demo.html` (the data inlined — a single
shareable file). The page's `1× / 2× / 4× / 8×` buttons control playback speed;
`8×` runs the whole ~5-minute session in ~40 s. Source template:
`scripts/gating_demo_template.html`.

---

## ⚠️ Hardware honesty (read before recording)

The MPU6050 is a **±2 g MEMS accelerometer** (~400 µg/√Hz), roughly **10⁶× less
sensitive** than a broadband seismometer, and it measures **acceleration**. It
**cannot** detect real teleseismic earthquakes or reproduce a real P-then-S
waveform. What the live sensor genuinely demonstrates is that the full causal
preprocessing + inference + transmission-gating pipeline runs **in real time on
edge hardware** and that a **physical vibration triggers it**. Real P-then-S phase
picking is shown **only** via `--source replay`, which streams real STEAD
earthquake waveforms through the identical pipeline.

**Tap vs. shake:** a *sustained shake* is more model-like than a single sharp tap
(the model keys on a sustained-energy onset envelope, not a real P coda). We do
**not** pre-claim tap-vs-shake selectivity — the console reports what actually
fired. On synthetic/live accelerometer input the model is **not** tap-selective (a
sharp broadband tap triggers both gates). The selectivity result that carries the
thesis — **3.08 vs 17.51 false-triggers/hr** (`outputs/causal/run.json`) — is
measured on **real STEAD data**, and is reproduced live only in `replay`.

---

## Wiring (Raspberry Pi ↔ MPU6050, I²C)

| MPU6050 pin | Pi pin |
|---|---|
| VCC | 3.3 V (pin 1) |
| GND | GND (pin 6) |
| SDA | GPIO2 / SDA1 (pin 3) |
| SCL | GPIO3 / SCL1 (pin 5) |

Enable I²C (`sudo raspi-config` → Interface Options → I²C), then confirm the
sensor is on the bus at address **0x68**:

```bash
sudo apt install -y i2c-tools
i2cdetect -y 1        # expect 0x68 in the grid
```

The reader configures ±2 g (`AFS_SEL=0`) and an on-chip **100 Hz** output rate
(`SMPLRT_DIV=9` with DLPF). **Axis convention** (a labeling choice, *not* a
calibration): `az→Z, ax→N, ay→E`.

## Install

On a Raspberry Pi (live-inference only — **no torch/seisbench needed**):

```bash
pip install -r requirements-edge.txt   # numpy scipy onnxruntime pyyaml matplotlib smbus2
```

Or by hand:

```bash
pip install onnxruntime numpy scipy matplotlib pyyaml
pip install smbus2          # only needed for --source mpu6050 (live I²C)
```

If `smbus2` or the I²C device is unavailable, `--source mpu6050` prints a notice
and **falls back to `smoke`** — it never crashes.

---

## Running

Sensible defaults: the causal INT8 model is `outputs/demo/causal_stage3_int8.onnx`
and the replay bundle is `outputs/demo/replay_traces.npz`, so the common cases are
just:

```bash
python scripts/demo_edge.py --source smoke      # synthetic, ~90 s
python scripts/demo_edge.py --source replay     # real STEAD, 5 traces
python scripts/demo_edge.py --source mpu6050     # live sensor (Pi)
```

Useful flags:

- `--plot` — live scrolling waveform + STA/LTA and model alarm lanes + byte
  readout (needs a display; on a headless box it writes a static preview to
  `outputs/demo/plot_preview.png` instead of crashing).
- `--speed N` — playback speed for smoke/replay (default 4× real time); `--fast`
  disables pacing entirely (for quick checks).
- `--duration-s N` — live `mpu6050` capture length (default 90 s).

Every run writes the two deliverables:

- `outputs/demo/bytes_report.json` — per-gate transmit counts, false counts, byte totals.
- `outputs/demo/bytes_comparison.png` — **the headline three-bar chart** (raw ≫ STA/LTA > model).

---

## The three-act recording script

**Act 1 — live, quiet room** (`--source mpu6050`, sit still).
Tap the table a few times. STA/LTA fires on the taps; the model mostly stays
silent. As the session runs, the byte counters **diverge** — this is the low
false-trigger-rate, robustness story. STA/LTA firing on a sharp tap is expected
and fine to show.

**Act 2 — live, sustained shake** (`--source mpu6050`, shake the sensor for a few
seconds). The model **fires and transmits** — it is selective, not deaf. STA/LTA
also fires here; that is fine. This shows a physical vibration driving the full
real-time edge pipeline.

**Act 3 — replay real STEAD** (`--source replay`). For each real earthquake the
model transmits an event packet and recovers **P and S** within ~100–300 ms of the
catalog arrivals (`✓ model recovered: P … S …`), while the pure-noise trace stays
silent on **both** gates. This is the "as-envisioned" proof — labeled clearly as
**replayed real seismic data**, not the accelerometer.

**Close** on `outputs/demo/bytes_comparison.png`: raw continuous transmission is
enormous, STA/LTA gating is medium (inflated by false alarms), model gating is
small. Same real events, a fraction of the bytes.

> Report what the run actually shows. On synthetic/live accelerometer input the
> model is not tap-selective; the false-trigger-rate win is a real-STEAD result,
> reproduced live in `replay`.

---

## How it works (six stages)

1. **Source → raw (3, N) @ 100 Hz.** MEMS I²C, synthetic, or real STEAD.
2. **Causal preprocess + inference.** Live: a rolling 6000-sample buffer,
   warm-started on **real ambient** (never zeros), causally preprocessed each hop
   and run through the causal INT8 ONNX model. Replay: `causal_stream_probabilities`
   (the validated streaming-equivalent path). STA/LTA runs each hop on demeaned data.
3. **Transmission gating.** On a rising-edge trigger (model detection ≥ 0.80, or
   STA/LTA ratio ≥ 3.0), one event packet is serialized (3 s pre + 7 s post clip +
   picks + label) and its real byte size measured. A 12 s per-gate refractory means
   one packet per event. Three cumulative counters: raw-continuous (every digitized
   sample × 3 ch × 2 B), STA/LTA-gated, model-gated. Triggers outside a known event
   window are counted as **false** transmissions (scripted sources only).
4. **Views.** Console (default, SSH-friendly) or `--plot`.
5. **Replay** as above.
6. **Session end.** `bytes_report.json` + the three-bar `bytes_comparison.png`.

## Regenerating the checked-in artifacts (needs STEAD + the checkpoint, on the VM)

```bash
# real causal FP32 -> INT8 ONNX (STEAD-calibrated)
python scripts/export_onnx.py   --causal --checkpoint checkpoints/stage3_causal/best.pt \
    --out outputs/demo/causal_stage3.onnx
python scripts/quantize_onnx.py --causal --checkpoint checkpoints/stage3_causal/best.pt \
    --fp32-onnx outputs/demo/causal_stage3.onnx \
    --int8-out  outputs/demo/causal_stage3_int8.onnx \
    --report    outputs/demo/causal_stage3_quant_report.json --eval-limit 400
# real STEAD replay bundle (only traces the model picks cleanly are kept)
python scripts/build_replay_bundle.py
```
