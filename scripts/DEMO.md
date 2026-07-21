# Live edge demo — MPU3050/MPU6050 transmission gating

`scripts/demo_edge.py` runs the project's causal INT8 phase-picker as an on-device
**transmission gate** for a bandwidth-limited remote sensor, and compares it head
to head with a classic STA/LTA energy detector. The headline is **bytes on the
wire**: STA/LTA false-triggers on any energy burst and transmits far more; the
learned model is selective and transmits far less for the same real events.

Four source modes feed one identical pipeline:

| `--source` | what it is | shows |
|---|---|---|
| `smoke`   | deterministic synthetic accel stream (no hardware, no dataset) | both gates behave; byte counters diverge |
| `mpu3050` | interleaved I²C reads from a physical MPU3050 (Raspberry Pi) | the full pipeline runs online from the requested gyro hardware; motion can trigger it |
| `mpu6050` | interleaved I²C reads from a physical MPU6050 (Raspberry Pi) | the full pipeline runs online from acceleration samples; vibration can trigger it |
| `replay`  | **real STEAD earthquakes** streamed through the same causal path | genuine P **and** S phase picks the live motion sensors cannot produce |

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
`8×` runs the whole ~6-minute session in ~45 s. Source template:
`scripts/gating_demo_template.html`.

---

## ⚠️ Hardware honesty (read before recording)

The MPU6050 is a consumer MEMS accelerometer; the MPU3050 is a three-axis MEMS
**gyroscope**. Neither is a calibrated seismometer, and their units and transfer
functions differ from the ground-velocity traces used to train this model. The
live modes therefore demonstrate online acquisition, persistent causal
preprocessing, inference, and delayed packet transmission on a Pi — **not**
validated earthquake detection. Real P-then-S picking is shown only with
`--source replay`, which streams real STEAD waveforms through the same runtime.

**Tap vs. shake:** a *sustained shake* is more model-like than a single sharp tap
(the model keys on a sustained-energy onset envelope, not a real P coda). We do
**not** pre-claim tap-vs-shake selectivity — the console reports what actually
fired. On synthetic/live accelerometer input the model is **not** tap-selective (a
sharp broadband tap triggers both gates). The **3.08 vs 17.51
false-triggers/hr** result (`outputs/causal/run.json`) is a separate, curated
STEAD P-stream study. The edge demo gates on the detection stream, so do not
present those rates as a direct measurement of this demo.

---

## Wiring (Raspberry Pi ↔ MPU3050 or MPU6050, I²C)

| breakout pin | Pi pin |
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

Both readers configure a nominal **100 Hz** output (`SMPLRT_DIV=9` with DLPF).
The MPU6050 uses ±2 g; the MPU3050 uses ±250 °/s (`FS_SEL=0`) and reports gyro
X/Y/Z. Use a breakout that is safe at the Pi's 3.3 V logic level. Axis mapping is
only a channel-label convention, not a seismic calibration.

## Install

On a Raspberry Pi (live-inference only — **no torch/seisbench needed**):

```bash
pip install -r requirements-edge.txt   # numpy scipy onnxruntime pyyaml matplotlib smbus2
```

Or by hand:

```bash
pip install onnxruntime numpy scipy matplotlib pyyaml
pip install smbus2          # needed for either live I²C source
```

If `smbus2` or the I²C device is unavailable, a hardware run exits with an error
instead of silently becoming a synthetic demo. Add `--fallback-smoke` only when
you intentionally want that fallback.

---

## Running

Sensible defaults: the causal INT8 model is `outputs/demo/causal_stage3_int8.onnx`
and the replay bundle is `outputs/demo/replay_traces.npz`, so the common cases are
just:

```bash
python scripts/demo_edge.py --source smoke      # synthetic, ~90 s
python scripts/demo_edge.py --source replay      # real STEAD, 6 traces
python scripts/demo_edge.py --source mpu3050     # live gyro (Pi)
python scripts/demo_edge.py --source mpu6050     # live accelerometer (Pi)
```

Useful flags:

- `--plot` — live scrolling waveform + STA/LTA and model alarm lanes + byte
  readout (needs a display; on a headless box it writes a static preview to
  `outputs/demo/plot_preview.png` instead of crashing).
- `--speed N` — playback speed for smoke/replay (default 4× real time); `--fast`
  disables pacing entirely (for quick checks).
- `--duration-s N` — live acquisition length after ambient warm-up (default 90 s).
- `--i2c-bus N`, `--i2c-address 0x68` — select the bus and device address.
- `--fallback-smoke` — explicitly substitute the synthetic stream if hardware
  initialization fails.

Every run writes the two deliverables:

- `outputs/demo/bytes_report.json` — per-gate transmit counts, false counts, byte totals.
- `outputs/demo/bytes_comparison.png` — **the headline three-bar chart** (raw ≫ STA/LTA > model).

---

## The three-act recording script

**Act 1 — live, quiet room** (`--source mpu3050`, hold the sensor still).
Move or rotate it a few times and narrate only what actually fires. This proves
that acquisition and inference are interleaved on the Pi; it does not establish
seismic accuracy or tap-vs-shake selectivity.

**Act 2 — live, sustained motion** (`--source mpu3050`, rotate or shake the
sensor for a few seconds). If a gate triggers, point out the two states: the
decision is immediate, then transmission waits until the full 7 s post-trigger
window exists. The packet always contains exactly 3 s before and 7 s after.

**Act 3 — replay real STEAD** (`--source replay`). For each real earthquake the
model transmits an event packet and recovers **P and S** within ~100–300 ms of the
catalog arrivals (`✓ model recovered: P … S …`), while the pure-noise trace stays
silent on **both** gates. This is the "as-envisioned" proof — labeled clearly as
**replayed real seismic data**, not the accelerometer.

**Close** on `outputs/demo/bytes_comparison.png`: in the bundled replay, the same
compressed-NPZ wire format produces 36 continuous packets, 7 STA/LTA packets,
and 4 model packets. The conclusion is fewer transmitted packets, not a codec
comparison.

> Report what the run actually shows. On synthetic/live accelerometer input the
> model is not tap-selective; the false-trigger-rate win is a real-STEAD result,
> reproduced live in `replay`.

---

## How it works (six stages)

1. **Source → raw (3, N) @ 100 Hz.** MEMS I²C, synthetic, or real STEAD.
2. **Causal preprocess + inference.** Hardware reads, preprocessing, and
   inference are interleaved one hop at a time. A persistent preprocessor sees
   every new sample exactly once; the rolling model window is warm-started on
   real ambient rather than zeros. Replay uses the validated
   `causal_stream_probabilities` path. STA/LTA runs each hop on demeaned data.
3. **Transmission gating.** On a rising-edge trigger (model detection ≥ 0.80, or
   STA/LTA ratio ≥ 3.0), transmission is scheduled for 7 s later. Only then is a
   fixed 10 s packet serialized (3 s pre + 7 s post + picks + label). A 12 s
   per-gate refractory means one packet per event. Continuous and gated paths use
   the same int16 compressed-NPZ packet codec; uncompressed payload bytes are also
   reported separately. Triggers outside a known event window are counted as
   **false** transmissions for scripted sources.
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
