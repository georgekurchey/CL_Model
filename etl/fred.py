from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

RAW = Path("data/raw/fred"); RAW.mkdir(parents=True, exist_ok=True)

def _seed(series: list[str]) -> None:
    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    for s in series:
        df = pd.DataFrame({"date": dates, s: 100.0}).astype({s: "float"})
        df.to_csv(RAW / f"{s}.csv", index=False)

def main(series: list[str], since: str = "2015-01-01") -> None:
    _seed(series)
    for s in series:
        print(str(RAW / f"{s}.csv"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", type=lambda x: x.split(","), default=["DTWEXBGS","DGS10","T10YIE"])
    ap.add_argument("--since", default="2015-01-01")
    args = ap.parse_args()
    main(args.series, args.since)
