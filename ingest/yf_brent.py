import argparse, os
from pathlib import Path
import pandas as pd
import numpy as np
import datetime as dt
import requests

# ---------- Yahoo helpers ----------
def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        try: df = df.droplevel(0, axis=1)
        except Exception: df.columns = [str(c) for c in df.columns]
    ren = {"Adj Close":"adj_close","AdjClose":"adj_close","adj close":"adj_close","Close":"close"}
    df = df.rename(columns=ren)
    for c in list(df.columns):
        if c.lower() in ("adj_close","close","open","high","low","price","last","px"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _pick_price(df: pd.DataFrame) -> pd.Series|None:
    for c in ("adj_close","close","Adj Close","Close","price","last","px","open"):
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            s = df[c].dropna()
            if s.size >= 5 and s.var() > 0: return df[c]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for c in num_cols:
        s = df[c].dropna()
        if s.size >= 5 and s.var() > 0: return df[c]
    return None

def _download_yahoo(ticker: str, start_date: dt.date):
    import yfinance as yf
    df = yf.download(ticker, start=start_date, interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty: return None
    df = _normalize_df(df)
    px = _pick_price(df)
    if px is None: return None
    out = px.to_frame(name="brent_close").reset_index()
    if "Date" in out.columns: out = out.rename(columns={"Date":"date"})
    elif "index" in out.columns: out = out.rename(columns={"index":"date"})
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out = out.sort_values("date").dropna(subset=["brent_close"])
    return out

# ---------- EIA fallback (official Brent Europe spot, PET.RBRTE.D) ----------
def _download_eia(start: str):
    api_key = os.environ.get("EIA_API_KEY","")
    if not api_key: return None
    url = f"https://api.eia.gov/series/?api_key={api_key}&series_id=PET.RBRTE.D"
    r = requests.get(url, timeout=20)
    if r.status_code != 200: return None
    js = r.json()
    if "series" not in js or not js["series"]: return None
    data = js["series"][0].get("data", [])
    if not data: return None
    df = pd.DataFrame(data, columns=["date_str","brent_close"])
    df["date"] = pd.to_datetime(df["date_str"])
    df = df[["date","brent_close"]]
    df["brent_close"] = pd.to_numeric(df["brent_close"], errors="coerce")
    df = df[df["date"] >= pd.to_datetime(start)]
    df = df.sort_values("date").dropna()
    return df if not df.empty else None

# ---------- FRED fallback (DCOILBRENTEU) ----------
def _download_fred(start: str):
    api_key = os.environ.get("FRED_API_KEY","")
    if not api_key: return None
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id":"DCOILBRENTEU","api_key":api_key,"file_type":"json","observation_start":start}
    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200: return None
    obs = r.json().get("observations", [])
    if not obs: return None
    df = pd.DataFrame(obs)
    if df.empty: return None
    df = df[["date","value"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["brent_close"] = pd.to_numeric(df["value"].replace(".", None), errors="coerce")
    df = df.drop(columns=["value"]).dropna().sort_values("date")
    return df if not df.empty else None

def fetch_brent_any(tickers: list[str], start: str) -> pd.DataFrame:
    sdt = dt.datetime.strptime(start, "%Y-%m-%d").date()
    # Try Yahoo list
    for tk in tickers:
        try:
            df = _download_yahoo(tk, sdt)
            if df is not None and not df.empty:
                df = df[["date","brent_close"]]
                return df
        except Exception:
            pass
    # EIA fallback
    df = _download_eia(start)
    if df is not None and not df.empty:
        return df[["date","brent_close"]]
    # FRED fallback
    df = _download_fred(start)
    if df is not None and not df.empty:
        return df[["date","brent_close"]]
    raise SystemExit("Brent download failed from Yahoo; EIA and FRED fallbacks unavailable or empty.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="BZ=F")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data/external/brent.csv"))
    args = ap.parse_args()
    try_list = [args.ticker]
    for alt in ("BZ=F","CO=F"): 
        if alt not in try_list: try_list.append(alt)
    df = fetch_brent_any(try_list, args.start)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote Brent -> {args.out} rows={len(df)} latest={df['date'].max().date()}")
if __name__ == "__main__":
    main()
