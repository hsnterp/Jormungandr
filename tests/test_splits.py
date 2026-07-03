import numpy as np
import pandas as pd

from seismic_edge_picker import splits as S


def _synthetic_metadata(n_events=200, traces_per_event=3, n_noise=150):
    rows = []
    for e in range(n_events):
        for _ in range(traces_per_event):
            rows.append({
                S.COL_CATEGORY: "earthquake_local",
                S.COL_SOURCE: f"evt{e}",
                S.COL_NETWORK: "XX",
                S.COL_STATION: f"ST{e % 40}",
                S.COL_TRACE: f"eq_{e}_{np.random.randint(1e9)}",
                S.COL_SNR: "[10.0 12.0 8.0]",
            })
    for k in range(n_noise):
        rows.append({
            S.COL_CATEGORY: "noise",
            S.COL_SOURCE: None,
            S.COL_NETWORK: "XX",
            S.COL_STATION: f"NS{k % 30}",
            S.COL_TRACE: f"noise_{k}",
            S.COL_SNR: None,
        })
    return pd.DataFrame(rows)


def test_no_event_leakage_across_splits():
    meta = _synthetic_metadata()
    sp = S.make_splits(meta, ratios=(0.8, 0.1, 0.1), by="event", hash_mod=1000)
    groups = {}
    for name, idx in sp.items():
        groups[name] = set(S.group_key(meta.iloc[i], "event") for i in idx)
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])


def test_split_covers_all_rows():
    meta = _synthetic_metadata()
    sp = S.make_splits(meta)
    total = sum(len(v) for v in sp.values())
    assert total == len(meta)


def test_split_ratios_roughly_hold():
    meta = _synthetic_metadata(n_events=500, traces_per_event=2, n_noise=400)
    sp = S.make_splits(meta, ratios=(0.8, 0.1, 0.1))
    frac_train = len(sp["train"]) / len(meta)
    assert 0.7 < frac_train < 0.9


def test_select_subset_balanced():
    meta = _synthetic_metadata(n_events=300, traces_per_event=3, n_noise=300)
    idx = S.select_subset(meta, n_earthquake=100, n_noise=50, seed=1)
    cat = meta.iloc[idx][S.COL_CATEGORY].astype(str)
    assert (cat == "earthquake_local").sum() == 100
    assert (cat == "noise").sum() == 50


def test_select_subset_deterministic():
    meta = _synthetic_metadata()
    a = S.select_subset(meta, 50, 25, seed=7)
    b = S.select_subset(meta, 50, 25, seed=7)
    assert np.array_equal(a, b)


def test_parse_snr_db():
    assert abs(S.parse_snr_db("[10.0 12.0 8.0]") - 10.0) < 1e-6
    assert abs(S.parse_snr_db("[10, 20, 30]") - 20.0) < 1e-6
    assert np.isnan(S.parse_snr_db(None))
    assert abs(S.parse_snr_db(5.0) - 5.0) < 1e-6
