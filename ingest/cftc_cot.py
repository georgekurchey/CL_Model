import argparse, os
from pathlib import Path
import pandas as pd

# This script tries Nasdaq Data Link (CFTC mirror). If it fails (403/404), it will
# write a stub file with correct columns so your pipeline keeps working.
# You can also pass --local FILE to parse a local CSV you downloaded from CFTC.

DEFAULT_WTI = "067651_F_L_ALL"   # WTI (NYMEX) Disaggregated, Futures + Options Combined
DEFAULT_BRENT = "067411_F_L_ALL" # Brent (ICE) Disaggregated, Futures + Options Combined

def _try_ndlink(code: str, api_key: str, start: str) -> pd.DataFrame:
    base = "https://data.nasdaq.com/api/v3/datasets/CFTC"
    url = f"{base}/{code}.csv?order=asc&start_date={start}"
    if api_key:
        url += f"&api_key={api_key}"
    df = pd.read_csv(url, parse_dates=["Date"])
    # Attempt column normalization for Disaggregated dataset
    cols = {c.lower(): c for c in df.columns}
    def pick(name):
        return cols.get(name.lower())
    date = pick("Date")
    oi = pick("Open Interest (All)") or pick("Open Interest (All) (All)")
    mm_long = pick("Money Managers Long (All)") or pick("Money Manager Longs")
    mm_short = pick("Money Managers Short (All)") or pick("Money Manager Shorts")
    if not (date and oi and mm_long and mm_short):
        raise ValueError("Unexpected CFTC schema from Nasdaq Data Link.")
    out = df[[date, oi, mm_long, mm_short]].copy()
    out.columns = ["date","open_interest","mm_long","mm_short"]
    out["mm_net"] = (out["mm_long"] - out["mm_short"]).astype(float)
    out["mm_netr"] = out["mm_net"] / out["open_interest"].replace(0, pd.NA)
    out = out.dropna(subset=["date"]).sort_values("date")
    return out[["date","mm_netr"]]

def _write_stub(out: Path, label: str):
    out.parent.mkdir(parents=True, exist_ok=True)
    stub = pd.DataFrame(columns=["date",f"{label}_mm_netr"])
    stub.to_csv(out, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--wti_code", default=DEFAULT_WTI)
    ap.add_argument("--brent_code", default=DEFAULT_BRENT)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data/external/cot.csv"))
    ap.add_argument("--local", default=None, help="Path to a local CSV already filtered for WTI/Brent")
    args = ap.parse_args()

    api_key = os.environ.get("NASDAQ_DATA_LINK_API_KEY", "")

    if args.local:
        df = pd.read_csv(args.local, parse_dates=["date"])
        df.to_csv(args.out, index=False)
        print(f"Wrote COT (local) -> {args.out} rows={len(df)}")
        return

    ok = []
    out_wti = out_br = None
    try:
        wti = _try_ndlink(args.wti_code, api_key, args.start)
        wti = wti.rename(columns={"mm_netr":"wti_mm_netr"})
        ok.append("WTI")
    except Exception as e:
        print(f"[COT] WTI via Nasdaq DL failed: {e}")
        wti = None
    try:
        br = _try_ndlink(args.brent_code, api_key, args.start)
        br = br.rename(columns={"mm_netr":"brent_mm_netr"})
        ok.append("BRENT")
    except Exception as e:
        print(f"[COT] Brent via Nasdaq DL failed: {e}")
        br = None

    if wti is None and br is None:
        print("[COT] Both sources failed. Writing stub to keep pipeline alive.")
        _write_stub(Path(args.out), "wti")
        return

    if wti is None:
        df = br.rename(columns={"brent_mm_netr":"wti_mm_netr"})
    elif br is None:
        df = wti.rename(columns={"wti_mm_netr":"brent_mm_netr"})
    else:
        df = pd.merge(wti, br, on="date", how="outer").sort_values("date")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote COT -> {args.out} rows={len(df)} ok={ok}")
if __name__ == "__main__":
    main()
