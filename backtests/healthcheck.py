import sys, os, time
from pathlib import Path

# thresholds (seconds)
DAY = 24*3600
THRESHOLDS = {
    # inputs
    "data/raw/eia/weekly_crude_stocks.csv": 8*DAY,
    "data/raw/fred/DTWEXBGS.csv":           3*DAY,
    "data/raw/fred/DGS10.csv":              3*DAY,
    "data/raw/fred/T10YIE.csv":             3*DAY,
    # optional inputs (only warn if missing)
    "data/raw/cftc/managed_money.parquet":  14*DAY,
    "data/raw/cvol/wti_cvol.parquet":       14*DAY,
    # features
    "data/proc/features.parquet":           1*DAY,
    # reports
    "reports/report.txt":                   1*DAY
}

WARN_ONLY = {
    "data/raw/cftc/managed_money.parquet",
    "data/raw/cvol/wti_cvol.parquet",
}

def age_seconds(p: Path, now: float) -> float:
    try:
        return now - p.stat().st_mtime
    except FileNotFoundError:
        return float('inf')

def main() -> int:
    now = time.time()
    errors = []
    warnings = []
    for rel, thresh in THRESHOLDS.items():
        p = Path(rel)
        a = age_seconds(p, now)
        if not p.exists():
            msg = f"MISSING: {rel}"
            (warnings if rel in WARN_ONLY else errors).append(msg)
            continue
        if a > thresh:
            days = round(a / DAY, 2)
            limit = round(thresh / DAY, 2)
            msg = f"STALE: {rel} age={days}d > limit={limit}d"
            (warnings if rel in WARN_ONLY else errors).append(msg)

    for w in warnings:
        print("[WARN]", w)
    for e in errors:
        print("[ERROR]", e)

    if errors:
        print(f"\nHealthcheck FAILED: {len(errors)} error(s).", file=sys.stderr)
        return 1
    print("Healthcheck OK")
    return 0

if __name__ == "__main__":
    sys.exit(main())
