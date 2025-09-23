from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

RAW = Path("data/raw/eia"); RAW.mkdir(parents=True, exist_ok=True)

def _example_weekly() -> pd.DataFrame:
    idx = pd.date_range("2020-01-03", periods=10, freq="W-FRI")
    return pd.DataFrame({"date": idx, "us_crude_stocks": range(10)}).astype({"us_crude_stocks": "float"})

def main(since: str = "2015-01-01") -> None:
    f = RAW / "weekly_crude_stocks.csv"
    _example_weekly().to_csv(f, index=False)
    print(str(f))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2015-01-01")
    args = ap.parse_args()
    main(since=args.since)
