"""Unit tests for src/cleaning.py. Run with: python -m pytest"""

import numpy as np
import pandas as pd

from src.cleaning import add_numpy_features, clean_demand, fill_short_gaps


def make_raw(values, start="2024-01-01"):
    periods = pd.date_range(start, periods=len(values), freq="h").strftime("%Y-%m-%dT%H")
    return pd.DataFrame({"period": periods, "demand_mwh": [str(v) for v in values]})


def test_fills_short_gaps_but_not_long_ones():
    s = pd.Series([1.0, np.nan, np.nan, 4.0, 5.0, np.nan, np.nan, np.nan, np.nan, np.nan, 11.0])
    out = fill_short_gaps(s, max_gap=3)
    assert out.iloc[1] == 2.0 and out.iloc[2] == 3.0  # 2-hour gap filled
    assert out.iloc[5:10].isna().all()                # 5-hour gap left alone, not partly invented


def test_parses_string_values_from_eia():
    clean, report = clean_demand(make_raw([40000] * 48))
    assert clean["demand_mwh"].dtype == float
    assert report["raw_rows"] == 48


def test_drops_duplicate_hours():
    raw = make_raw([40000] * 48)
    raw = pd.concat([raw, raw.iloc[[5]]], ignore_index=True)
    clean, report = clean_demand(raw)
    assert report["duplicate_hours"] == 1
    assert len(clean) == 48


def test_reindexes_missing_hours():
    raw = make_raw([40000] * 48).drop(index=[10, 11])  # two hours missing from the feed
    clean, report = clean_demand(raw)
    assert report["missing_hours"] == 2
    assert len(clean) == 48
    assert clean["demand_mwh"].notna().all()  # short gap was filled


def test_removes_zero_and_spike_outliers():
    values = [40000] * 168  # one week, so each hour has same-hour neighbors on nearby days
    values[20] = 0          # impossible value
    values[84] = 120000     # 3x spike
    clean, report = clean_demand(make_raw(values))
    assert report["outliers_removed"] == 2
    assert clean["demand_mwh"].iloc[20] == 40000  # replaced by interpolation
    assert clean["demand_mwh"].iloc[84] == 40000


def test_keeps_normal_daily_peaks():
    hours = np.arange(24 * 14)
    values = (40000 + 15000 * np.sin(2 * np.pi * hours / 24)).round()  # big but normal daily swing
    _, report = clean_demand(make_raw(values))
    assert report["outliers_removed"] == 0


def test_numpy_features():
    df = pd.DataFrame({
        "hour": [0, 6, 12, 18],
        "day_of_year": [1, 2, 185, 185],
        "day_of_week": [1, 2, 6, 0],
        "temp_c": [5.0, 18.0, 30.0, 35.0],
        "temp_lag_24h": [10.0, 18.0, 25.0, 35.0],
        "lag_24h": [40000.0, 50000.0, 60000.0, 80000.0],
        "lag_168h": [40000.0, 40000.0, 60000.0, 80000.0],
        "avg_prior_week": [40000.0, 50000.0, 50000.0, 80000.0],
        "local_ts": ["2024-01-01 00:00:00", "2024-01-02 06:00:00", "2024-07-06 12:00:00", "2024-07-07 18:00:00"],
    })
    out = add_numpy_features(df)
    assert np.allclose(out["hour_sin"] ** 2 + out["hour_cos"] ** 2, 1.0)
    assert (out["cooling_degrees"] >= 0).all() and (out["heating_degrees"] >= 0).all()
    assert out.loc[0, "heating_degrees"] == 13.0 and out.loc[3, "cooling_degrees"] == 17.0
    assert out["temp_change_24h"].tolist() == [-5.0, 0.0, 5.0, 0.0]
    assert out.loc[0, "heating_change_24h"] == 5.0    # 13 heating degrees today vs. 8 yesterday
    assert out.loc[2, "cooling_change_24h"] == 5.0    # 12 cooling degrees today vs. 7 yesterday
    assert out["weekly_ratio"].tolist() == [1.0, 1.25, 1.0, 1.0]
    assert out.loc[2, "level_ratio"] == 1.2
    assert out["is_holiday"].tolist() == [1, 0, 0, 0]            # Jan 1 is New Year's Day
    assert out["is_holiday_yesterday"].tolist() == [0, 1, 0, 0]  # Jan 2 follows a holiday
    assert out["is_weekend"].tolist() == [0, 0, 1, 1]
    assert out["is_weekend_yesterday"].tolist() == [1, 0, 0, 1]  # Monday and Sunday follow weekend days
