import io, sys, argparse
from pathlib import Path
import pandas as pd, requests

def from_local_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        raise SystemExit("CSV must contain a 'date' column")
    for c in ("cvol_1m","skew_25d","term_premium"):
        if c not in df.columns:
            df[c] = pd.NA
    out = df[["date","cvol_1m","skew_25d","term_premium"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date")
    return out

def from_fred_ovx() -> pd.DataFrame:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=OVXCLS"
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        sys.stderr.write(f"FRED HTTP {r.status_code}\n")
        raise SystemExit(1)
    raw = pd.read_csv(io.BytesIO(r.content))
    cols_lower = {c.lower(): c for c in raw.columns}
    if "date" in cols_lower:
        date_col = cols_lower["date"]
    elif "observation_date" in cols_lower:
        date_col = cols_lower["observation_date"]
    else:
        raise SystemExit(f"Unexpected FRED CSV schema: {raw.columns.tolist()}")
    val_col = None
    for c in raw.columns:
        if c != date_col:
            val_col = c
            break
    if val_col is None:
        raise SystemExit(f"No value column found: {raw.columns.tolist()}")
    out = pd.DataFrame({
        "date": pd.to_datetime(raw[date_col], errors="coerce"),
        "cvol_1m": pd.to_numeric(raw[val_col], errors="coerce"),
        "skew_25d": pd.NA,
        "term_premium": pd.NA
    })
    out = out.dropna(subset=["date"]).sort_values("date")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--proxy-ovx", action="store_true")
    args = ap.parse_args()
    if args.csv:
        df = from_local_csv(args.csv)
    elif args.proxy_ovx:
        df = from_fred_ovx()
    else:
        raise SystemExit("Specify --csv PATH or --proxy-ovx")
    Path("data/raw/cvol").mkdir(parents=True, exist_ok=True)
    df.to_parquet("data/raw/cvol/wti_cvol.parquet", index=False)
    print("data/raw/cvol/wti_cvol.parquet", len(df))

if __name__ == "__main__":
    main()
