import os
import sys
from pathlib import Path
import pandas as pd
import nasdaqdatalink as ndl

def fetch(code: str, api_key: str, start: str = None):
    ndl.ApiConfig.api_key = api_key
    try:
        df = ndl.get(code, start_date=start)
    except Exception as e:
        sys.stderr.write(f"NASDAQ/CHRIS fetch skipped ({type(e).__name__}: {e})\n")
        return None
    df = df.reset_index().rename(columns={"Date": "date"})
    return df

def main():
    key = os.getenv("NASDAQ_API_KEY", "").strip()
    if not key:
        sys.stderr.write("NASDAQ_API_KEY not set; skipping CHRIS fetch.\n")
        return
    outdir = Path("data/raw/chris")
    outdir.mkdir(parents=True, exist_ok=True)
    codes = [f"CHRIS/NYMEX_CL{i}" for i in range(1, 13)]
    for i, code in enumerate(codes, start=1):
        df = fetch(code, key, start="2015-01-01")
        if df is None:
            continue
        keep = df[["date", "Settle"]].rename(columns={"Settle": "settle"})
        fn = outdir / f"CL{i}.csv"
        keep.to_csv(fn, index=False)
        print(fn)

if __name__ == "__main__":
    main()
