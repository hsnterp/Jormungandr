import numpy as np

from seismic_edge_picker.preprocessing import demean, bandpass, normalize


def test_demean_removes_offset():
    x = np.ones((3, 6000)) * 5.0 + np.random.randn(3, 6000)
    y = demean(x)
    assert np.allclose(y.mean(axis=-1), 0.0, atol=1e-5)
    assert y.dtype == np.float32


def test_bandpass_attenuates_dc_and_shape():
    fs = 100.0
    t = np.arange(6000) / fs
    x = 3.0 + np.sin(2 * np.pi * 10 * t)  # DC + 10 Hz (in-band)
    x = np.stack([x, x, x])
    y = bandpass(x, 1.0, 45.0, fs, order=4)
    assert y.shape == x.shape
    # DC component should be strongly attenuated
    assert abs(y.mean()) < 0.1


def test_bandpass_removes_out_of_band():
    fs = 100.0
    t = np.arange(6000) / fs
    inband = np.sin(2 * np.pi * 10 * t)
    outband = np.sin(2 * np.pi * 0.2 * t)  # below 1 Hz
    x = np.stack([inband + outband] * 3)
    y = bandpass(x, 1.0, 45.0, fs, order=4)
    # energy of filtered signal close to in-band-only energy
    assert np.std(y[0]) < np.std(x[0])


def test_normalize_std_unit():
    x = np.random.randn(3, 6000) * 7.0
    y = normalize(x, "std")
    assert abs(y.std() - 1.0) < 1e-3


def test_normalize_max():
    x = np.random.randn(3, 6000) * 7.0
    y = normalize(x, "max")
    assert abs(np.abs(y).max() - 1.0) < 1e-3


def test_normalize_handles_zero():
    x = np.zeros((3, 6000))
    y = normalize(x, "std")
    assert np.all(np.isfinite(y))
