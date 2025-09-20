import argparse, os, requests
from pathlib import Path
import pandas as pd

API = "https://api.stlouisfed.org/fred/series/observations"

def fetch_series(series_id: str, api_key: str, start: str) -> pd.DataFrame:
    params = {"series_id": series_id, "api_key": api_key, "file_type":"json", "observation_start": start}
    r = requests.get(API, params=params, timeout=15)
    r.raise_for_status()
    js = r.json()
    obs = js.get("observations", [])
    df = pd.DataFrame(obs)
    if df.empty:
        return pd.DataFrame(columns=["date", series_id])
    df = df[["date","value"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    # FRED uses "." for missing
    df["value"] = pd.to_numeric(df["value"].replace(".", None), errors="coerce")
    df = df.rename(columns={"value": series_id})
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1]/"config/fred_series.json"))
    ap.add_argument("--start", default=None)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]/"data/external/fred_macro.csv"))
    args = ap.parse_args()

    cfg = Path(args.config)
    if not cfg.exists():
        raise SystemExit(f"Config not found: {cfg}")
    js = __import__("json").loads(cfg.read_text())
    series_map = js.get("series", {})
    start = args.start or js.get("start", "2015-01-01")

    api_key = os.environ.get("FRED_API_KEY","")
    if not api_key:
        raise SystemExit("FRED_API_KEY missing (export or save in .env and load)")

    # fetch all and outer-join by date
    dfs = []
    for sid, col in series_map.items():
        df = fetch_series(sid, api_key, start)
        df = df.rename(columns={sid: col})
        dfs.append(df)
    from functools import reduce
    out = reduce(lambda l, r: pd.merge(l, r, on="date", how="outer"), dfs).sort_values("date")
    out.to_csv(args.out, index=False)
    print(f"Wrote FRED macro -> {args.out} cols={list(out.columns)} rows={len(out)}")
if __name__ == "__main__":
    main()
