import sys, argparse, os
from pathlib import Path
import pandas as pd
from datetime import datetime

DEF_PATH = "/Users/georgekurchey/CL_Model/data/raw/cl_strip.parquet"

def choose_path(p: Path) -> Path:
    if p.exists():
        return p
    # try common alternates
    p_csv = p.with_suffix(".csv")
    p_parq = p.with_suffix(".parquet")
    candidates = [q for q in [p, p_csv, p_parq] if q.exists()]
    if not candidates:
        raise SystemExit(f"File not found: {p} (also tried {p_csv} and {p_parq})")
    # pick the most recently modified
    return max(candidates, key=lambda q: q.stat().st_mtime)

def load_df(p: Path) -> pd.DataFrame:
    if p.suffix == ".parquet":
        try:
            import pyarrow  # noqa
            df = pd.read_parquet(p)
        except Exception as e:
            raise SystemExit(f"Failed to read parquet: {p}\n{e}")
    else:
        try:
            df = pd.read_csv(p, parse_dates=["date"])
        except Exception as e:
            raise SystemExit(f"Failed to read csv: {p}\n{e}")
    return df

def main():
    ap = argparse.ArgumentParser(description="Validate CL strip file (date, ticker, settle, source_code).")
    ap.add_argument("--path", default=DEF_PATH, help="Path to cl_strip.[parquet|csv]")
    ap.add_argument("--expect_m", type=int, default=12, help="Expected number of front-months on latest date")
    ap.add_argument("--show", type=int, default=5, help="Rows to show in head/tail preview")
    args = ap.parse_args()

    p = choose_path(Path(args.path))
    df = load_df(p)

    print(f"File: {p}  rows={len(df):,}")
    required = ["date","ticker","settle","source_code"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"ERROR: missing columns: {missing_cols}")

    # basic type normalization
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_date = df["date"].isna().sum()
    if bad_date > 0:
        raise SystemExit(f"ERROR: {bad_date} rows have invalid 'date'")

    # ensure tz-naive
    df["date"] = df["date"].dt.tz_localize(None)

    # settle numeric
    df["settle"] = pd.to_numeric(df["settle"], errors="coerce")
    bad_settle = df["settle"].isna().sum()
    if bad_settle > 0:
        print(f"WARNING: {bad_settle} rows have NaN 'settle' (will be dropped from checks)")

    # drop any full invalid rows from further checks
    dfx = df.dropna(subset=["date","settle"]).copy()

    # negative/zero settle
    nonpos = (dfx["settle"] <= 0).sum()
    if nonpos > 0:
        raise SystemExit(f"ERROR: {nonpos} rows have non-positive 'settle'")

    # duplicates per (date,ticker)
    dups = dfx.duplicated(subset=["date","ticker"]).sum()
    if dups > 0:
        print(f"WARNING: {dups} duplicate (date,ticker) rows found (keeping last).")
        dfx = dfx.sort_values(["date","ticker"]).drop_duplicates(["date","ticker"], keep="last")

    # per-ticker monotone dates
    bad_order = 0
    for tkr, g in dfx.groupby("ticker"):
        if not g["date"].is_monotonic_increasing:
            bad_order += 1
    if bad_order > 0:
        print(f"WARNING: {bad_order} tickers have non-monotonic dates (after de-dup).")

    # recency
    dt_max = dfx["date"].max()
    print(f"Date span: {dfx['date'].min().date()} → {dt_max.date()} (latest)")
    # coverage on latest date
    latest = dfx[dfx["date"] == dt_max]
    uniq = sorted(latest["ticker"].unique())
    print(f"Tickers on latest date ({dt_max.date()}): {uniq}")
    missing = [f"CL{i}" for i in range(1, args.expect_m+1) if f"CL{i}" not in uniq]
    if missing:
        print(f"WARNING: latest date missing {len(missing)}/{args.expect_m}: {', '.join(missing)}")
    else:
        print(f"OK: latest date has full CL1..CL{args.expect_m}")

    # quick per-ticker stats
    by_tkr = (
        dfx.groupby("ticker")
           .agg(rows=("settle","size"),
                first=("date","min"),
                last=("date","max"),
                min_settle=("settle","min"),
                max_settle=("settle","max"))
           .reset_index()
           .sort_values("ticker")
    )
    print("\nPer-ticker coverage (first/last date, rows):")
    print(by_tkr.to_string(index=False))

    # preview
    sh = max(1, args.show)
    print("\nHead:")
    print(dfx.sort_values(["date","ticker"]).head(sh).to_string(index=False))
    print("\nTail:")
    print(dfx.sort_values(["date","ticker"]).tail(sh).to_string(index=False))

    # exit code: 0 if passed critical checks
    if missing:
        # not fatal for Yahoo gaps, but return non-zero so automation can flag
        sys.exit(2)
    sys.exit(0)

if __name__ == "__main__":
    main()
