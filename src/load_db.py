"""Clean the raw CSVs with pandas and load them into a SQLite database.

Usage:
    python -m src.load_db

Creates data/gridsense.db with two tables:
    demand(ts, local_ts, demand_mwh)   one row per UTC hour, local_ts in Central time
    weather(ts, temp_c)
"""

import json
import sqlite3
from pathlib import Path

import pandas as pd

from src.cleaning import LOCAL_TZ, clean_demand, clean_weather

RAW_DIR = Path("data/raw")
DB_PATH = Path("data/gridsense.db")
RESULTS_DIR = Path("results")


def main() -> None:
    demand_raw = pd.read_csv(RAW_DIR / "ercot_demand.csv", dtype={"period": str})
    weather_raw = pd.read_csv(RAW_DIR / "texas_weather.csv")

    demand, report = clean_demand(demand_raw)
    weather = clean_weather(weather_raw)

    demand_table = pd.DataFrame({
        "ts": demand.index.strftime("%Y-%m-%d %H:%M:%S"),
        "local_ts": demand.index.tz_convert(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "demand_mwh": demand["demand_mwh"].to_numpy(),
    })
    weather_table = pd.DataFrame({
        "ts": weather.index.strftime("%Y-%m-%d %H:%M:%S"),
        "temp_c": weather["temp_c"].to_numpy(),
    })

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        demand_table.to_sql("demand", conn, if_exists="replace", index=False)
        weather_table.to_sql("weather", conn, if_exists="replace", index=False)
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_demand_ts ON demand(ts)")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_weather_ts ON weather(ts)")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "data_quality.json").write_text(json.dumps(report, indent=2))

    print("Data quality report:")
    for key, value in report.items():
        print(f"  {key}: {value:,}")
    print(f"Loaded {len(demand_table):,} demand rows and {len(weather_table):,} weather rows into {DB_PATH}")


if __name__ == "__main__":
    main()
