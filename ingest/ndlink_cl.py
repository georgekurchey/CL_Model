import os, time, argparse, datetime as dt, re, io
from pathlib import Path
from typing import Optional, List
import pandas as pd
import requests, certifi
import yfinance as yf

MONTH_CODES = ["F","G","H","J","K","M","N","Q","U","V","X","Z"]  # Jan..Dec

def _today_utc():
    # crude rolls in the month prior; we will start strip at next calendar month
    return dt.datetime.now(dt.UTC).date()

def _read_env_key() -> str:
    raw = os.environ.get("NASDAQ_DATA_LINK_API_KEY", "")
    raw = str(raw).strip().strip('"\'')

    if " " in raw:
        raw = raw.split()[-1]
    m = re.search(r"[A-Za-z0-9_-]{10,}", raw)
    if m:
        raw = m.group(0)
    return raw

def _http_get_csv(url: str, params: dict, insecure: bool, tries: int = 3, timeout: int = 30) -> pd.DataFrame:
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout,
                             verify=(False if insecure else certifi.where()))
            r.raise_for_status()
            return pd.read_csv(io.BytesIO(r.content), parse_dates=["Date"])
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(1.0 + k)

def _fetch_contract_nasdaq(n: int, api_key: str, start_date: Optional[str], sleep_s: float, insecure: bool) -> pd.DataFrame:
    code = f"CHRIS/CME_CL{n}"
    base = f"https://data.nasdaq.com/api/v3/datasets/{code}.csv"
    params = {"api_key": api_key, "order": "asc"}
    if start_date: params["start_date"] = start_date
    df = _http_get_csv(base, params, insecure=insecure)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date","ticker","settle","source_code"])
    df.columns = [c.lower().strip() for c in df.columns]
    if "settle" not in df.columns:
        if "last" in df.columns:
            df["settle"] = df["last"]
        else:
            raise ValueError(f"{code}: response missing 'settle'")
    df = df.rename(columns={"date":"date"})[["date","settle"]].copy()
    df["ticker"] = f"CL{n}"
    df["source_code"] = code
    df = df.dropna(subset=["date","settle"])
    df = df[df["settle"] > 0]
    df = df.sort_values("date").drop_duplicates(["date","ticker"], keep="last")
    time.sleep(max(0.0, sleep_s))
    return df

def _next_n_contract_symbols(n: int, ref_date: Optional[dt.date] = None) -> List[str]:
    """Return Yahoo symbols for next n WTI contracts, starting at NEXT calendar month."""
    if ref_date is None:
        ref_date = _today_utc()
    y = ref_date.year
    m = ref_date.month
    # start at next month (crude front rolls in prior month)
    m += 1
    if m > 12:
        m = 1
        y += 1
    out = []
    for i in range(n):
        mm_index = ((m - 1) + i) % 12
        year_offset = ((m - 1) + i) // 12
        code = MONTH_CODES[mm_index]
        yy = (y + year_offset) % 100
        sym = f"CL{code}{yy:02d}.NYM"
        out.append(sym)
    return out


def _fetch_contract_yahoo(idx: int, symbol: str, start_date: Optional[str], sleep_s: float) -> pd.DataFrame:
    """Fetch per-contract from Yahoo; use Close as settle; map to CL{idx}. Handle MultiIndex and empties safely."""
    try:
        df = yf.download(symbol, start=start_date, interval="1d",
                         progress=False, auto_adjust=False, threads=False)
    except Exception:
        df = None

    if df is None or df.empty:
        print(f"[WARN] Yahoo empty for {symbol}; skipping.")
        return pd.DataFrame(columns=["date","ticker","settle","source_code"])

    # Normalize to a Series 'settle'
    def extract_close(_df: pd.DataFrame) -> pd.Series | None:
        if isinstance(_df.columns, pd.MultiIndex):
            # Prefer the top-level 'Close'
            try:
                sub = _df['Close']
            except Exception:
                try:
                    sub = _df.xs('Close', axis=1, level=0)
                except Exception:
                    return None
            # sub could be Series or DataFrame (if multiple inner columns)
            if isinstance(sub, pd.DataFrame):
                if sub.shape[1] == 0:
                    return None
                ser = sub.iloc[:, 0]
            else:
                ser = sub
        else:
            if 'Close' not in _df.columns:
                return None
            ser = _df['Close']
        ser = ser.copy()
        ser.name = 'settle'
        return ser

    ser = extract_close(df)
    if ser is None or ser.empty:
        print(f"[WARN] Could not extract Close for {symbol}; skipping.")
        return pd.DataFrame(columns=["date","ticker","settle","source_code"])

    close = ser.reset_index()
    # yfinance usually labels the index column 'Date'
    if 'Date' in close.columns:
        close = close.rename(columns={'Date': 'date'})
    elif 'index' in close.columns:
        close = close.rename(columns={'index': 'date'})
    # Clean
    close['date'] = pd.to_datetime(close['date']).dt.tz_localize(None)
    close = close.dropna(subset=['date', 'settle'])
    close = close[close['settle'] > 0]

    close['ticker'] = f"CL{idx}"
    close['source_code'] = f"YF/{symbol}"
    close = close.sort_values('date').drop_duplicates(['date', 'ticker'], keep='last')

    time.sleep(max(0.0, sleep_s))
    return close[['date','settle','ticker','source_code']]


def _merge_incremental(existing: Optional[pd.DataFrame], fresh: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        out = fresh.copy()
    else:
        out = pd.concat([existing, fresh], axis=0, ignore_index=True)
        out = out.sort_values(["date","ticker"]).drop_duplicates(["date","ticker"], keep="last")
    return out

def _load_existing(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists(): return None
    if path.suffix == ".parquet":
        try:
            import pyarrow  # noqa
            return pd.read_parquet(path)
        except Exception: pass
    if path.suffix == ".csv":
        return pd.read_csv(path, parse_dates=["date"])
    p_csv = path.with_suffix(".csv"); p_parq = path.with_suffix(".parquet")
    if p_parq.exists():
        try:
            import pyarrow  # noqa
            return pd.read_parquet(p_parq)
        except Exception: pass
    if p_csv.exists():
        return pd.read_csv(p_csv, parse_dates=["date"])
    return None

def _save(df: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow  # noqa
        p = out_path.with_suffix(".parquet")
        df.to_parquet(p, index=False)
        return p
    except Exception:
        p = out_path.with_suffix(".csv")
        df.to_csv(p, index=False)
        return p

def _infer_start_date(existing: Optional[pd.DataFrame], backfill_years: int, overlap_days: int) -> Optional[str]:
    if existing is None or existing.empty:
        start = _today_utc() - dt.timedelta(days=365*backfill_years)
        return start.isoformat()
    max_date = pd.to_datetime(existing["date"]).max().date()
    start = max_date - dt.timedelta(days=overlap_days)
    return start.isoformat()

def _validate_complete_strip(df: pd.DataFrame, m: int) -> List[str]:
    if df.empty: return ["empty_dataframe"]
    latest = pd.to_datetime(df["date"]).max()
    dfl = df[pd.to_datetime(df["date"]) == latest]
    missing = []
    for n in range(1, m+1):
        if not any(dfl["ticker"] == f"CL{n}"):
            missing.append(f"CL{n}")
    return missing

def main():
    ap = argparse.ArgumentParser(description="Ingest CL futures strip from Nasdaq Data Link or Yahoo Finance.")
    ap.add_argument("--source", choices=["nasdaq","yahoo"], default="yahoo", help="Data source")
    ap.add_argument("--m", type=int, default=12, help="Number of contracts (front M).")
    ap.add_argument("--out", type=str, default="/Users/georgekurchey/CL_Model/data/raw/cl_strip.parquet", help="Output (parquet preferred).")
    ap.add_argument("--backfill_years", type=int, default=10)
    ap.add_argument("--overlap_days", type=int, default=10)
    ap.add_argument("--sleep", type=float, default=0.05)
    ap.add_argument("--start_date", type=str, default=None)
    ap.add_argument("--insecure", action="store_true", help="Disable SSL verification (Nasdaq only; not recommended).")
    args = ap.parse_args()

    out_path = Path(args.out)
    existing = _load_existing(out_path)
    start_date = args.start_date or _infer_start_date(existing, args.backfill_years, args.overlap_days)

    frames = []
    if args.source == "nasdaq":
        api_key = _read_env_key()
        if not api_key:
            raise SystemExit("NASDAQ_DATA_LINK_API_KEY not found. Try --source yahoo or set the key.")
        for n in range(1, args.m + 1):
            df_n = _fetch_contract_nasdaq(n, api_key, start_date, args.sleep, insecure=args.insecure)
            frames.append(df_n)
            print(f"[Nasdaq] CL{n}: {len(df_n):,} rows (from {start_date})")
    else:
        syms = _next_n_contract_symbols(args.m)
        missing_syms = []
        for idx, sym in enumerate(syms, start=1):
            df_n = _fetch_contract_yahoo(idx, sym, start_date, args.sleep)
            if df_n.empty:
                missing_syms.append(sym)
            frames.append(df_n)
            print(f"[Yahoo ] {sym} -> CL{idx}: {len(df_n):,} rows (from {start_date})")
        if missing_syms:
            print(f"[WARN] Missing/empty Yahoo symbols: {', '.join(missing_syms)}")

    fresh = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame(columns=["date","ticker","settle","source_code"])
    merged = _merge_incremental(existing, fresh)
    if not merged.empty:
        merged["date"] = pd.to_datetime(merged["date"]).dt.tz_localize(None)
        merged = merged.sort_values(["date","ticker"]).reset_index(drop=True)

    saved_path = _save(merged, out_path)
    print(f"Saved -> {saved_path}  (rows={len(merged):,})")

    missing = _validate_complete_strip(merged, args.m)
    if missing:
        print(f"WARNING: Latest date missing {len(missing)} tickers: {', '.join(missing)}")
    else:
        latest = pd.to_datetime(merged['date']).max()
        print(f"OK: Latest strip complete for {latest.date()} (CL1..CL{args.m})")

if __name__ == "__main__":
    main()
