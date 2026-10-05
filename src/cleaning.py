"""Data cleaning and feature engineering with pandas and NumPy.

Kept separate from the scripts so every function can be unit-tested (see tests/).
"""

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

LOCAL_TZ = "America/Chicago"  # ERCOT runs on Central time; demand follows local daily routines
BASE_TEMP_C = 18.0            # standard base temperature for heating/cooling degree calculations


def fill_short_gaps(series: pd.Series, max_gap: int = 3) -> pd.Series:
    """Linearly interpolate gaps of at most `max_gap` consecutive hours; leave longer gaps missing.

    pandas' interpolate(limit=n) would partially fill long gaps too, inventing data inside
    a multi-hour outage. Here we measure each gap's full length first and only fill short ones.
    """
    is_na = series.isna()
    run_id = (is_na != is_na.shift()).cumsum()
    run_len = is_na.groupby(run_id).transform("sum")
    fillable = is_na & (run_len <= max_gap)
    interpolated = series.interpolate(limit_area="inside")
    return series.where(~fillable, interpolated)


def clean_demand(raw: pd.DataFrame, max_gap: int = 3, outlier_ratio: float = 0.5) -> tuple[pd.DataFrame, dict]:
    """Clean raw EIA rows (period, demand_mwh) into a complete hourly UTC series.

    Steps: parse types, drop duplicate hours, reindex so every hour exists, null out
    impossible values (<= 0) and spikes far from the same hour on nearby days, then fill
    short gaps. Returns the cleaned frame plus a data-quality report.
    """
    df = raw.copy()
    df["ts"] = pd.to_datetime(df["period"], format="%Y-%m-%dT%H", utc=True)
    df["demand_mwh"] = pd.to_numeric(df["demand_mwh"], errors="coerce").astype(float)  # EIA returns strings

    report = {"raw_rows": len(df), "duplicate_hours": int(df.duplicated("ts").sum())}
    df = df.drop_duplicates("ts").set_index("ts").sort_index()

    full_range = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
    report["missing_hours"] = int(len(full_range) - len(df))
    s = df["demand_mwh"].reindex(full_range)

    # Compare each hour to the SAME hour on the 3 days before and after. A normal afternoon peak
    # sits far above the overnight trough, so a plain rolling median would flag healthy peaks.
    same_hour_neighbors = pd.concat([s.shift(24 * k) for k in (-3, -2, -1, 1, 2, 3)], axis=1)
    same_hour_median = same_hour_neighbors.median(axis=1).where(same_hour_neighbors.count(axis=1) >= 3)
    deviation = (s - same_hour_median).abs() / same_hour_median
    bad = (s <= 0) | (deviation > outlier_ratio)
    report["outliers_removed"] = int(bad.sum())
    s = s.mask(bad)

    s = fill_short_gaps(s, max_gap)
    report["still_missing_after_fill"] = int(s.isna().sum())

    out = s.to_frame("demand_mwh")
    out.index.name = "ts"
    return out, report


def clean_weather(raw: pd.DataFrame, max_gap: int = 3) -> pd.DataFrame:
    """Average the cities' hourly temperatures into one statewide series (UTC index)."""
    df = raw.copy()
    df["ts"] = pd.to_datetime(df["time"], utc=True)
    temp = df.groupby("ts")["temp_c"].mean().sort_index()
    full_range = pd.date_range(temp.index.min(), temp.index.max(), freq="h", tz="UTC")
    temp = fill_short_gaps(temp.reindex(full_range), max_gap)
    out = temp.to_frame("temp_c")
    out.index.name = "ts"
    return out


def add_numpy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features computed in NumPy to the SQL feature table.

    - Cyclical encodings: hour 23 and hour 0 are neighbors, which raw integers hide.
    - Cooling/heating degrees: demand rises with heat (AC) and with cold (heating), not linearly.
    - Day-over-day changes: how much hotter/colder than yesterday, and whether yesterday was a
      weekend or holiday. These drive the "ratio to yesterday" models in train.py.
    - Ratio features: recent demand relative to recent demand, so they don't depend on the
      grid's overall size (which keeps growing).
    """
    out = df.copy()
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["doy_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)

    out["cooling_degrees"] = np.maximum(out["temp_c"] - BASE_TEMP_C, 0.0)
    out["heating_degrees"] = np.maximum(BASE_TEMP_C - out["temp_c"], 0.0)
    cooling_yesterday = np.maximum(out["temp_lag_24h"] - BASE_TEMP_C, 0.0)
    heating_yesterday = np.maximum(BASE_TEMP_C - out["temp_lag_24h"], 0.0)
    out["temp_change_24h"] = out["temp_c"] - out["temp_lag_24h"]
    out["cooling_change_24h"] = out["cooling_degrees"] - cooling_yesterday
    out["heating_change_24h"] = out["heating_degrees"] - heating_yesterday

    out["weekly_ratio"] = out["lag_24h"] / out["lag_168h"]        # yesterday vs. same day last week
    out["level_ratio"] = out["lag_24h"] / out["avg_prior_week"]   # yesterday vs. last week's average

    local_dates = pd.to_datetime(out["local_ts"]).dt.normalize()
    yesterday = local_dates - pd.Timedelta(days=1)
    holidays = USFederalHolidayCalendar().holidays(yesterday.min(), local_dates.max())
    out["is_holiday"] = local_dates.isin(holidays).astype(int)
    out["is_holiday_yesterday"] = yesterday.isin(holidays).astype(int)
    out["is_weekend"] = out["day_of_week"].isin([0, 6]).astype(int)  # SQLite %w: 0 = Sunday, 6 = Saturday
    out["is_weekend_yesterday"] = out["day_of_week"].isin([0, 1]).astype(int)  # yesterday was Sat or Sun
    return out
