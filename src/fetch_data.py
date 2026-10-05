"""Download hourly ERCOT electricity demand (EIA) and Texas temperatures (Open-Meteo).

Usage:
    export EIA_API_KEY=your_key_here          # free key: https://www.eia.gov/opendata/
    python -m src.fetch_data --start 2023-01-01 --end 2026-09-27

Writes:
    data/raw/ercot_demand.csv   columns: period, demand_mwh   (period = UTC hour, "YYYY-MM-DDTHH")
    data/raw/texas_weather.csv  columns: time, city, temp_c   (time = UTC hour)
"""

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path("data/raw")

EIA_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
EIA_PAGE_SIZE = 5000  # the EIA API returns at most 5,000 rows per request

WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
# ERCOT's largest load centers. Their average temperature is a good proxy for statewide demand.
CITIES = {
    "houston": (29.76, -95.37),
    "dallas": (32.78, -96.80),
    "san_antonio": (29.42, -98.49),
    "austin": (30.27, -97.74),
}


def fetch_eia_demand(api_key: str, start: str, end: str) -> pd.DataFrame:
    """Page through the EIA API and return every hourly ERCOT demand value in [start, end]."""
    rows, offset = [], 0
    while True:
        params = {
            "api_key": api_key,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": "ERCO",  # ERCOT balancing authority
            "facets[type][]": "D",           # D = demand
            "start": f"{start}T00",
            "end": f"{end}T23",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": EIA_PAGE_SIZE,
        }
        resp = requests.get(EIA_URL, params=params, timeout=60)
        resp.raise_for_status()
        batch = resp.json()["response"]["data"]
        rows.extend(batch)
        print(f"  EIA: {len(rows):,} rows so far")
        if len(batch) < EIA_PAGE_SIZE:
            break
        offset += EIA_PAGE_SIZE
        time.sleep(0.5)  # be polite to the API

    if not rows:
        raise RuntimeError("EIA returned no data. Check your API key and date range.")
    df = pd.DataFrame(rows)
    return df[["period", "value"]].rename(columns={"value": "demand_mwh"})


def fetch_weather(start: str, end: str) -> pd.DataFrame:
    """Hourly 2 m air temperature (deg C, UTC) for each city from the Open-Meteo archive (no key needed)."""
    frames = []
    for city, (lat, lon) in CITIES.items():
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "hourly": "temperature_2m",
            "timezone": "GMT",
        }
        resp = requests.get(WEATHER_URL, params=params, timeout=60)
        resp.raise_for_status()
        hourly = resp.json()["hourly"]
        frames.append(pd.DataFrame({"time": hourly["time"], "city": city, "temp_c": hourly["temperature_2m"]}))
        print(f"  Weather: {city} done")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default="2023-01-01", help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD (weather archive lags ~5 days, so use a date at least a week ago)")
    args = parser.parse_args()

    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        raise SystemExit("Set the EIA_API_KEY environment variable first (free at https://www.eia.gov/opendata/).")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading ERCOT demand from EIA...")
    demand = fetch_eia_demand(api_key, args.start, args.end)
    demand.to_csv(RAW_DIR / "ercot_demand.csv", index=False)

    print("Downloading Texas weather from Open-Meteo...")
    weather = fetch_weather(args.start, args.end)
    weather.to_csv(RAW_DIR / "texas_weather.csv", index=False)

    print(f"Done: {len(demand):,} demand rows, {len(weather):,} weather rows in {RAW_DIR}/")


if __name__ == "__main__":
    main()
