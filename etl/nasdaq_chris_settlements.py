from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
from etl.utils import safe_write_csv, utc_now_iso

RAW = Path("data/raw/chris"); RAW.mkdir(parents=True, exist_ok=True)

def _seed(days: int = 120):
    import numpy as np
    dates = pd.bdate_range("2020-01-01", periods=days)
    for depth in range(1, 13):
        base = 70 + 0.2*depth
        settle = base + 0.03*np.arange(len(dates))
        df = pd.DataFrame({"date": dates, "settle": settle})
        safe_write_csv(df, RAW / f"CL{depth}.csv",
                       {"source":"seed","dataset":f"CL{depth}","vintage_datetime_utc": utc_now_iso()})

def fetch(code: str, api_key: str, start: str = "2015-01-01") -> pd.DataFrame:
    import nasdaqdatalink as ndl
    ndl.ApiConfig.api_key = api_key
    df = ndl.get(code, start_date=start)
    if "Settle" not in df.columns:
        raise RuntimeError(f"{code}: Settle missing")
    df = df.reset_index().rename(columns={"Date":"date","Settle":"settle"})
    return df[["date","settle"]].sort_values("date")

def main():
    seed = os.getenv("SEED_CME","").strip()
    key = os.getenv("NASDAQ_API_KEY","").strip()
    if not key and not seed:
        raise SystemExit("NASDAQ_API_KEY not set")
    if seed:
        _seed()
        for d in range(1,13):
            p = RAW / f"CL{d}.csv"
            if p.exists(): print(p)
        return
    paths = []
    for d in range(1,13):
        code = f"CHRIS/NYMEX_CL{d}"
        df = fetch(code, key, start="2015-01-01")
        meta = {"source":"Nasdaq Data Link (CHRIS)","dataset":code,"vintage_datetime_utc":utc_now_iso()}
        paths.append(safe_write_csv(df, RAW / f"CL{d}.csv", meta))
    for p in paths:
        print(p)

if __name__ == "__main__":
    main()
