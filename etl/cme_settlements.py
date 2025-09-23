from __future__ import annotations
import os, io, requests
from pathlib import Path
import pandas as pd
from etl.utils import safe_write_csv, utc_now_iso

RAW = Path("data/raw/cme"); RAW.mkdir(parents=True, exist_ok=True)
BASE = "https://data.nasdaq.com/api/v3/datasets"  # formerly Quandl

def fetch_chris_csv(code: str, api_key: str, start: str = "2015-01-01") -> pd.DataFrame:
    url = f"{BASE}/{code}.csv"
    r = requests.get(url, params={"api_key": api_key, "start_date": start}, timeout=30)
    r.raise_for_status()
    # Read CSV from memory; CHRIS has 'Date' and 'Settle' columns
    df = pd.read_csv(io.StringIO(r.text))
    if "Date" not in df.columns:
        raise RuntimeError(f"{code}: 'Date' column missing")
    if "Settle" not in df.columns:
        raise RuntimeError(f"{code}: 'Settle' column missing")
    df = df.rename(columns={"Date": "date", "Settle": "settle"})
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "settle"]].sort_values("date")

def main():
    key = os.getenv("NASDAQ_API_KEY", "").strip()
    if not key:
        raise SystemExit("NASDAQ_API_KEY is not set; export it before running.")
    paths = []
    for depth in range(1, 13):
        code = f"CHRIS/NYMEX_CL{depth}"
        df = fetch_chris_csv(code, key, start="2015-01-01")
        meta = {"source": "Nasdaq Data Link (CHRIS)", "dataset": code, "vintage_datetime_utc": utc_now_iso()}
        paths.append(safe_write_csv(df, RAW / f"CL{depth}.csv", meta))
    for p in paths:
        print(p)

if __name__ == "__main__":
    main()
