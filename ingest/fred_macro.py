import os, argparse, datetime as dt
from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd
import requests

ROOT = Path("/Users/georgekurchey/CL_Model")
DEF_OUT = ROOT / "data" / "raw" / "macro.parquet"

API = "https://api.stlouisfed.org/fred/series/observations"

def _read_key() -> str:
    k = os.environ.get("FRED_API_KEY", "").strip().strip('"').strip("'")
    if not k:
        raise SystemExit("FRED_API_KEY not set. Run: source /Users/georgekurchey/CL_Model/config/secrets.env")
    return k

def _load_existing(p: Path) -> Optional[pd.DataFrame]:
    if not p.exists():
        p_csv = p.with_suffix(".csv")
        if p_csv.exists():
            return pd.read_csv(p_csv, parse_dates=["date"])
        return None
    try:
        import pyarrow  # noqa
        return pd.read_parquet(p)
    except Exception:
        return pd.read_csv(p.with_suffix(".csv"), parse_dates=["date"])

def _save(df: pd.DataFrame, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow  # noqa
        p = out.with_suffix(".parquet")
        df.to_parquet(p, index=False)
        return p
    except Exception:
        p = out.with_suffix(".csv")
        df.to_csv(p, index=False)
        return p

def _infer_start(existing: Optional[pd.DataFrame], backfill_years: int, overlap_days: int) -> str:
    if existing is None or existing.empty:
        start = dt.date.today() - dt.timedelta(days=365*backfill_years)
        return start.isoformat()
    last = pd.to_datetime(existing["date"]).max().date()
    start = last - dt.timedelta(days=overlap_days)
    return start.isoformat()

def _fetch_series(sid: str, api_key: str, start_date: str) -> pd.DataFrame:
    params = {
        "series_id": sid,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date
    }
    r = requests.get(API, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    obs = js.get("observations", [])
    rows = []
    for o in obs:
        v = o.get("value", ".")
        if v == "." or v is None or v == "":
            continue
        try:
            val = float(v)
        except Exception:
            continue
        rows.append((o["date"], sid.lower(), val))
    df = pd.DataFrame(rows, columns=["date","series_id","value"])
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.sort_values(["date","series_id"]).drop_duplicates(["date","series_id"], keep="last")
    return df

def _merge(existing: Optional[pd.DataFrame], fresh: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        return fresh.copy()
    out = pd.concat([existing, fresh], axis=0, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out = out.sort_values(["date","series_id"]).drop_duplicates(["date","series_id"], keep="last")
    return out

def main():
    ap = argparse.ArgumentParser(description="Ingest FRED series (long tidy): date, series_id, value")
    ap.add_argument("--series", nargs="+", default=["DTWEXBGS","DGS10","T10YIE","DFII10"], help="FRED series IDs")
    ap.add_argument("--out", default=str(DEF_OUT))
    ap.add_argument("--backfill_years", type=int, default=15)
    ap.add_argument("--overlap_days", type=int, default=5)
    args = ap.parse_args()

    api_key = _read_key()
    outp = Path(args.out)
    existing = _load_existing(outp)
    start = _infer_start(existing, args.backfill_years, args.overlap_days)

    frames = []
    for sid in args.series:
        df = _fetch_series(sid, api_key, start)
        print(f"Fetched {sid}: {len(df):,} rows from {start}")
        frames.append(df)
    fresh = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame(columns=["date","series_id","value"])

    merged = _merge(existing, fresh)
    saved = _save(merged, outp)
    print(f"Saved -> {saved}  (rows={len(merged):,})")

    # Quick sanity: show last available date per series
    if not merged.empty:
        last = (
            merged.groupby("series_id")["date"]
            .max()
            .reset_index()
            .sort_values("series_id")
        )
        print("\nLatest per series:")
        for _, r in last.iterrows():
            print(f"  {r['series_id']}: {r['date'].date()}")

if __name__ == "__main__":
    main()
