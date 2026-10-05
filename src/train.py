"""Train and evaluate day-ahead ERCOT demand forecasts with scikit-learn.

Usage:
    python -m src.train --test-days 90

Compares two simple baselines against machine learning models trained two ways:
  * "predicts demand level":  the model outputs megawatt-hours directly.
  * "predicts ratio to yesterday": the model outputs tomorrow's demand as a ratio of the
    same hour today (e.g., 1.03 = 3% higher), then we multiply back by today's value.
ERCOT demand keeps growing, so the test period contains record highs the level models never
saw in training. Tree models can't predict above their training range; the ratio framing
fixes that because a "3% hotter-day bump" looks the same at 50 GW or 90 GW.

Splits are by time (train on the past, test on the most recent `test-days` days), never shuffled.
Writes results/metrics.csv and results/forecast_week.png.
"""

import argparse
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.cleaning import add_numpy_features

DB_PATH = Path("data/gridsense.db")
SQL_PATH = Path("sql/features.sql")
RESULTS_DIR = Path("results")

CATEGORICAL = ["hour", "day_of_week", "month"]
CALENDAR = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "is_holiday", "is_weekend"]

# Features for models that predict the demand level directly (the first attempt).
LEVEL_NUMERIC = ["temp_c", "cooling_degrees", "heating_degrees", "lag_24h", "lag_168h", "avg_prior_week", *CALENDAR]

# Scale-free features for models that predict the ratio to yesterday: no raw megawatt values.
RATIO_NUMERIC = [
    "temp_c", "cooling_degrees", "heating_degrees",
    "temp_change_24h", "cooling_change_24h", "heating_change_24h",
    "weekly_ratio", "level_ratio",
    "is_holiday_yesterday", "is_weekend_yesterday", *CALENDAR,
]


def load_features() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(SQL_PATH.read_text(), conn, parse_dates=["ts"])
    df = add_numpy_features(df)
    before = len(df)
    needed = ["target", *set(LEVEL_NUMERIC + RATIO_NUMERIC)]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=needed).reset_index(drop=True)
    print(f"Feature table: {len(df):,} usable rows ({before - len(df):,} dropped for missing values or lag warm-up)")
    return df


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def ridge(numeric: list[str]):
    """Linear model: needs one-hot calendar features and scaled numbers."""
    return make_pipeline(
        ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), numeric),
        ]),
        Ridge(alpha=1.0),
    )


def boosting():
    """Gradient-boosted trees: learn nonlinear effects (e.g., heat x time of day) on their own."""
    return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=42)


# (name, model, numeric features, framing)
EXPERIMENTS = [
    ("Ridge (predicts demand level)", ridge(LEVEL_NUMERIC), LEVEL_NUMERIC, "level"),
    ("Gradient boosting (predicts demand level)", boosting(), LEVEL_NUMERIC, "level"),
    ("Ridge (predicts ratio to yesterday)", ridge(RATIO_NUMERIC), RATIO_NUMERIC, "ratio"),
    ("Gradient boosting (predicts ratio to yesterday)", boosting(), RATIO_NUMERIC, "ratio"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-days", type=int, default=90)
    args = parser.parse_args()

    df = load_features()
    cutoff = df["ts"].max() - pd.Timedelta(days=args.test_days)
    train, test = df[df["ts"] <= cutoff], df[df["ts"] > cutoff]
    print(f"Train: {train['ts'].min():%Y-%m-%d} to {train['ts'].max():%Y-%m-%d} ({len(train):,} hours)")
    print(f"Test:  {test['ts'].min():%Y-%m-%d} to {test['ts'].max():%Y-%m-%d} ({len(test):,} hours)")
    print(f"Highest demand in training: {train['target'].max():,.0f} MWh | in test: {test['target'].max():,.0f} MWh")

    y_test = test["target"].to_numpy()
    predictions = {
        "Baseline: same hour yesterday": test["lag_24h"].to_numpy(),
        "Baseline: same hour last week": test["lag_168h"].to_numpy(),
    }
    for name, model, numeric, framing in EXPERIMENTS:
        features = CATEGORICAL + numeric
        if framing == "level":
            model.fit(train[features], train["target"])
            predictions[name] = model.predict(test[features])
        else:
            model.fit(train[features], train["target"] / train["lag_24h"])
            predictions[name] = model.predict(test[features]) * test["lag_24h"].to_numpy()

    rows = [
        {"model": name, "MAE_MWh": mean_absolute_error(y_test, pred), "MAPE_pct": mape(y_test, pred)}
        for name, pred in predictions.items()
    ]
    metrics = pd.DataFrame(rows).sort_values("MAE_MWh").reset_index(drop=True)
    best_baseline = metrics[metrics["model"].str.startswith("Baseline")]["MAE_MWh"].min()
    metrics["improvement_vs_best_baseline_pct"] = (1 - metrics["MAE_MWh"] / best_baseline) * 100

    RESULTS_DIR.mkdir(exist_ok=True)
    metrics.round(2).to_csv(RESULTS_DIR / "metrics.csv", index=False)
    print("\nTest-set results (lower error is better):")
    print(metrics.round(2).to_string(index=False))

    # Plot the final test week: actual vs. the best ML model vs. the best baseline.
    best_model = metrics[~metrics["model"].str.startswith("Baseline")].iloc[0]["model"]
    best_base = metrics[metrics["model"].str.startswith("Baseline")].iloc[0]["model"]
    week = (test["ts"] > test["ts"].max() - pd.Timedelta(days=7)).to_numpy()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(test.loc[week, "ts"], y_test[week], label="Actual", linewidth=2.2, color="black")
    ax.plot(test.loc[week, "ts"], predictions[best_model][week], label=best_model, linewidth=1.6, color="tab:blue")
    ax.plot(test.loc[week, "ts"], predictions[best_base][week], label=best_base,
            linewidth=1.2, linestyle="--", color="tab:orange")
    ax.set_title("ERCOT day-ahead demand forecast, final test week")
    ax.set_ylabel("Demand (MWh)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "forecast_week.png", dpi=150)
    print(f"\nSaved {RESULTS_DIR / 'metrics.csv'} and {RESULTS_DIR / 'forecast_week.png'}")


if __name__ == "__main__":
    main()
