import numpy as np

from seismic_edge_picker.labels import gaussian_bump, build_label_mask


def test_gaussian_bump_peaks_at_center():
    b = gaussian_bump(6000, 3000, sigma_samples=25)
    assert np.argmax(b) == 3000
    assert abs(b[3000] - 1.0) < 1e-6
    assert b.dtype == np.float32


def test_label_shapes_and_streams():
    y = build_label_mask(
        6000, p_sample=2000, s_sample=3500, coda_end_sample=4500,
        sampling_rate=100, pick_sigma_s=0.25, detection_source="coda",
    )
    assert y.shape == (3, 6000)
    # detection window between P and coda
    assert y[0, 2000:4500].min() == 1.0
    assert y[0, :2000].max() == 0.0
    assert y[0, 4500:].max() == 0.0
    # pick peaks at arrival samples
    assert np.argmax(y[1]) == 2000
    assert np.argmax(y[2]) == 3500


def test_noise_all_zero():
    y = build_label_mask(6000, p_sample=None, s_sample=None)
    assert y.shape == (3, 6000)
    assert np.count_nonzero(y) == 0


def test_nan_arrivals_treated_as_missing():
    y = build_label_mask(6000, p_sample=float("nan"), s_sample=float("nan"))
    assert np.count_nonzero(y) == 0


def test_detection_fallback_without_coda():
    # no coda -> use S + fallback
    y = build_label_mask(
        6000, p_sample=1000, s_sample=1500, coda_end_sample=None,
        sampling_rate=100, detection_source="coda", detection_fallback_s=5.0,
    )
    # window from P(1000) to S(1500)+500 = 2000
    assert y[0, 1000] == 1.0
    assert y[0, 1999] == 1.0
    assert y[0, 2000] == 0.0


def test_missing_s_only_p():
    y = build_label_mask(6000, p_sample=2000, s_sample=None, coda_end_sample=3000)
    assert np.argmax(y[1]) == 2000
    assert np.count_nonzero(y[2]) == 0
