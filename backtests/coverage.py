from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import kstest

TAUS = np.arange(0.05,0.96,0.05)
QCOLS = [f"q{int(t*100):02d}" for t in TAUS]

def var_coverage(y, q, tau):
    thr = q
    hits = (y < thr).astype(int)
    rate = hits.mean()
    n = len(hits)
    se = np.sqrt(tau*(1-tau)/max(n,1))
    return float(rate), float(se)

def pit(y, qs):
    y = np.asarray(y).reshape(-1,1)
    return (y <= qs).mean(axis=1)

def main():
    feat = pd.read_parquet("data/proc/features.parquet")
    y = feat["ret_1d"].astype(float).to_numpy()
    rows = []
    for p in Path("preds").glob("*.parquet"):
        df = pd.read_parquet(p)
        df = df.merge(feat[["date","ret_1d"]], on="date", how="inner")
        y2 = df["ret_1d"].to_numpy()
        qs = df[QCOLS].to_numpy()
        pitv = pit(y2, qs)
        ks = kstest(pitv, "uniform").pvalue
        i05 = QCOLS.index("q05"); i95 = QCOLS.index("q95")
        c05,_ = var_coverage(y2, qs[:,i05], 0.05)
        c95,_ = var_coverage(y2, qs[:,i95], 0.95)  # right tail implied
        rows.append({"model": p.stem, "PIT_KS_p": ks, "VaR05_cov": c05, "VaR95_cov": c95})
    out = pd.DataFrame(rows).sort_values("model")
    Path("reports").mkdir(parents=True, exist_ok=True)
    out.to_csv("reports/coverage.csv", index=False)
    print("reports/coverage.csv")
if __name__ == "__main__":
    main()
