import numpy as np

from seismic_edge_picker.augment import (
    random_window_shift,
    mix_noise,
    channel_dropout,
)


def test_window_shift_moves_arrivals():
    rng = np.random.default_rng(0)
    x = np.zeros((3, 6000), dtype=np.float32)
    x[:, 3000] = 1.0
    arr = {"p": 3000, "s": 3200, "coda": 3500}
    for _ in range(20):
        y, na = random_window_shift(x, dict(arr), max_shift_s=15.0,
                                    sampling_rate=100, rng=rng)
        assert y.shape == x.shape
        if na["p"] is not None:
            # the impulse should now sit at the shifted arrival index
            assert y[0, na["p"]] == 1.0


def test_window_shift_keeps_picks_in_bounds():
    rng = np.random.default_rng(1)
    x = np.zeros((3, 6000), dtype=np.float32)
    # picks near the edges must never be shifted out of the window
    arr = {"p": 5, "s": 5990, "coda": None}
    for _ in range(200):
        _, na = random_window_shift(x, dict(arr), max_shift_s=15.0,
                                    sampling_rate=100, rng=rng)
        assert na["p"] is not None and 0 <= na["p"] < 6000
        assert na["s"] is not None and 0 <= na["s"] < 6000


def test_window_shift_coda_may_leave_window():
    rng = np.random.default_rng(4)
    x = np.zeros((3, 6000), dtype=np.float32)
    # coda beyond the window is allowed to drop to None (detection uses fallback)
    arr = {"p": 100, "s": 200, "coda": 5995}
    seen_none = False
    for _ in range(200):
        _, na = random_window_shift(x, dict(arr), max_shift_s=15.0,
                                    sampling_rate=100, rng=rng)
        if na["coda"] is None:
            seen_none = True
    assert seen_none


def test_mix_noise_target_snr():
    rng = np.random.default_rng(2)
    sig = rng.standard_normal((3, 6000)).astype(np.float32)
    noise = rng.standard_normal((3, 6000)).astype(np.float32)
    mixed = mix_noise(sig, noise, target_snr_db=6.0)
    added = mixed - sig
    sig_p = np.mean(sig.astype(np.float64) ** 2)
    noise_p = np.mean(added.astype(np.float64) ** 2)
    snr = 10 * np.log10(sig_p / noise_p)
    assert abs(snr - 6.0) < 0.5


def test_mix_noise_zero_noise_is_noop():
    sig = np.random.randn(3, 6000).astype(np.float32)
    out = mix_noise(sig, np.zeros_like(sig), 0.0)
    assert np.allclose(out, sig)


def test_channel_dropout_zeros_channels_but_not_all():
    rng = np.random.default_rng(3)
    x = np.ones((3, 6000), dtype=np.float32)
    y = channel_dropout(x, max_channels=1, rng=rng)
    zeroed = [c for c in range(3) if np.all(y[c] == 0)]
    assert len(zeroed) <= 1
    # at least one channel survives
    assert any(np.any(y[c] != 0) for c in range(3))
