import os, argparse, datetime as dt, io, re
from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd
import requests

ROOT = Path("/Users/georgekurchey/CL_Model")
DEF_OUT = ROOT / "data" / "raw" / "eia.parquet"

# Weekly stocks (kbbl): crude ex SPR, total gasoline, total distillate
DEFAULT_SERIES = ["WCESTUS1", "WGTSTUS1", "WDISTUS1"]
API_V2_ROUTE = "https://api.eia.gov/v2/petroleum/stoc/wstk/data/"

def read_key() -> str:
  k = os.environ.get("EIA_API_KEY", "").strip().strip('"').strip("'")
  if not k:
    raise SystemExit("EIA_API_KEY not set. source /Users/georgekurchey/CL_Model/config/secrets.env")
  return k

def load_existing(p: Path) -> Optional[pd.DataFrame]:
  if not p.exists():
    p_csv = p.with_suffix(".csv")
    return pd.read_csv(p_csv, parse_dates=["date"]) if p_csv.exists() else None
  try:
    import pyarrow  # noqa
    return pd.read_parquet(p)
  except Exception:
    return pd.read_csv(p.with_suffix(".csv"), parse_dates=["date"])

def save(df: pd.DataFrame, out: Path) -> Path:
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

def infer_start(existing: Optional[pd.DataFrame], backfill_years: int, overlap_weeks: int) -> str:
  if existing is None or existing.empty:
    start = dt.date.today() - dt.timedelta(days=365*backfill_years)
    return start.isoformat()
  last = pd.to_datetime(existing["date"]).max().date()
  start = last - dt.timedelta(weeks=overlap_weeks)
  return start.isoformat()

def normalize_series(s: str) -> str:
  """
  Accepts 'WCESTUS1' or legacy 'PET.WCESTUS1.W' and returns 'WCESTUS1'.
  """
  u = s.strip().upper()
  # common pattern: PET.SERIES.W
  parts = u.split(".")
  if len(parts) >= 3 and parts[0] == "PET" and parts[-1] == "W":
    return parts[1]
  # drop trailing '.W' if present
  if u.endswith(".W"):
    u = u[:-2]
  return u

def fetch_v2(series_code: str, api_key: str, start_iso: Optional[str]) -> pd.DataFrame:
  rows: List[Dict] = []
  params = {
    "api_key": api_key,
    "frequency": "weekly",
    "data[0]": "value",
    "facets[series][]": series_code,  # short code like WCESTUS1
    "sort[0][column]": "period",
    "sort[0][direction]": "asc",
    "length": 5000,
    "offset": 0
  }
  if start_iso:
    params["start"] = start_iso

  total = None
  while True:
    r = requests.get(API_V2_ROUTE, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    resp = js.get("response", {})
    data = resp.get("data", [])
    if total is None:
      total = int(resp.get("total", len(data)))
    rows.extend(data)
    if len(rows) >= total or not data:
      break
    params["offset"] += params["length"]

  if not rows:
    return pd.DataFrame(columns=["date","series_id","level"])

  df = pd.DataFrame(rows)
  if "period" not in df.columns or "value" not in df.columns:
    return pd.DataFrame(columns=["date","series_id","level"])
  df = df[["period","value"]].copy()
  df["date"] = pd.to_datetime(df["period"], errors="coerce").dt.tz_localize(None)
  df["series_id"] = series_code.lower()
  df["level"] = pd.to_numeric(df["value"], errors="coerce")
  df = df.dropna(subset=["date","level"])
  df = df.sort_values(["date","series_id"]).drop_duplicates(["date","series_id"], keep="last")
  return df[["date","series_id","level"]]

def fetch_excel(series_code: str) -> pd.DataFrame:
  """
  Best-effort Excel fallback. EIA usually hosts .../hist_xls/<CODE>_w.xlsx
  """
  candidates = [
    f"https://www.eia.gov/dnav/pet/hist_xls/{series_code}_w.xlsx",
    f"https://www.eia.gov/dnav/pet/hist_xls/{series_code}_w.xls",
  ]
  last_err = None
  for url in candidates:
    try:
      r = requests.get(url, timeout=30)
      r.raise_for_status()
      # Prefer openpyxl; if not installed, this will raise
      try:
        df0 = pd.read_excel(io.BytesIO(r.content), sheet_name=0, header=None, engine="openpyxl")
      except Exception:
        # try without specifying engine (might work for .xls if xlrd<2 is installed)
        df0 = pd.read_excel(io.BytesIO(r.content), sheet_name=0, header=None)
      # find header row containing 'Date'
      hdr_row = None
      for i in range(min(10, len(df0))):
        vals = [str(v).strip().lower() for v in df0.iloc[i].tolist()]
        if any(v == "date" for v in vals):
          hdr_row = i
          break
      if hdr_row is None:
        hdr_row = 2
      df = pd.read_excel(io.BytesIO(r.content), sheet_name=0, header=hdr_row)
      date_col = next((c for c in df.columns if str(c).strip().lower() == "date"), None)
      if date_col is None:
        continue
      # value column: first non-date numeric column
      val_col = next((c for c in df.columns if c != date_col), None)
      if val_col is None:
        continue
      out = df[[date_col, val_col]].rename(columns={date_col:"date", val_col:"level"})
      out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
      out["level"] = pd.to_numeric(out["level"], errors="coerce")
      out["series_id"] = series_code.lower()
      out = out.dropna(subset=["date","level"])
      out = out.sort_values(["date","series_id"]).drop_duplicates(["date","series_id"], keep="last")
      return out[["date","series_id","level"]]
    except Exception as e:
      last_err = e
      continue
  # nothing worked
  print(f"[WARN] Excel fallback unavailable for {series_code}: {last_err}")
  return pd.DataFrame(columns=["date","series_id","level"])

def merge_incremental(existing: Optional[pd.DataFrame], fresh: pd.DataFrame) -> pd.DataFrame:
  if existing is None or existing.empty:
    return fresh.copy()
  out = pd.concat([existing, fresh], axis=0, ignore_index=True)
  out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
  out = out.sort_values(["date","series_id"]).drop_duplicates(["date","series_id"], keep="last")
  return out

def main():
  ap = argparse.ArgumentParser(description="Ingest EIA weekly stocks (API v2 + Excel fallback).")
  ap.add_argument("--series", nargs="+", default=DEFAULT_SERIES,
                  help="Accepts short codes (WCESTUS1) or legacy PET.WCESTUS1.W")
  ap.add_argument("--out", default=str(DEF_OUT))
  ap.add_argument("--backfill_years", type=int, default=15)
  ap.add_argument("--overlap_weeks", type=int, default=6)
  ap.add_argument("--since", type=str, default=None)
  args = ap.parse_args()

  api_key = read_key()
  outp = Path(args.out)
  existing = load_existing(outp)
  start_iso = infer_start(existing, args.backfill_years, args.overlap_weeks)

  frames: List[pd.DataFrame] = []
  for raw_sid in args.series:
    code = normalize_series(raw_sid)
    # Try API v2 first
    try:
      df = fetch_v2(code, api_key, start_iso)
    except Exception as e:
      print(f"[WARN] API v2 failed for {code}: {e}; trying Excel fallback.")
      df = pd.DataFrame(columns=["date","series_id","level"])
    if df.empty:
      df = fetch_excel(code)
      if start_iso and not df.empty:
        df = df[df["date"] >= pd.to_datetime(start_iso)]
    print(f"Fetched {code}: {len(df):,} rows (since {start_iso})")
    frames.append(df)

  fresh = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame(columns=["date","series_id","level"])
  merged = merge_incremental(existing, fresh)
  if args.since:
    merged = merged[merged["date"] >= pd.to_datetime(args.since)]
  saved = save(merged, outp)
  print(f"Saved -> {saved}  (rows={len(merged):,})")

  if not merged.empty:
    last = merged.groupby("series_id")["date"].max().reset_index().sort_values("series_id")
    print("\nLatest per series:")
    for _, r in last.iterrows():
      print(f"  {r['series_id']}: {r['date'].date()}")

if __name__ == "__main__":
  main()
