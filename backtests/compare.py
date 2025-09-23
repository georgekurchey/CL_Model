from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from .metrics import crps_from_quantiles, pit_values

TAUS = np.arange(0.05,0.96,0.05)
QC = [f"q{int(t*100):02d}" for t in TAUS]

def load_preds():
    preds = {}
    for p in Path("preds").glob("*.parquet"):
        preds[p.stem] = pd.read_parquet(p)
    return preds

def align(y, P):
    keys = sorted(P.keys())
    base = None
    for k in keys:
        df = P[k]
        if base is None: base = df[["date"]].copy()
        else: base = base.merge(df[["date"]], on="date", how="inner")
    return base["date"]

def main():
    feat = pd.read_parquet("data/proc/features.parquet")
    P = load_preds()
    if not P: 
        raise SystemExit("no preds/*.parquet found")
    dates = align(feat, P)
    feat = feat.merge(dates.to_frame("date"), on="date", how="inner")
    y = feat["ret_1d"].values
    rows = []
    for name, df in sorted(P.items()):
        df = df.merge(dates.to_frame("date"), on="date", how="inner")
        qhat = df[QC].to_numpy()
        crps = crps_from_quantiles(y, qhat, TAUS)
        pit = float(pit_values(y, qhat, TAUS).mean())
        rows.append({"model":name, "CRPS":crps, "PIT_mean":pit})
    out = pd.DataFrame(rows).sort_values("CRPS")
    Path("reports").mkdir(parents=True, exist_ok=True)
    out.to_csv("reports/compare.csv", index=False)
    print("reports/compare.csv")
if __name__ == "__main__":
    main()
