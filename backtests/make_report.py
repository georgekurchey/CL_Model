from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from .scoring import pinball

def main():
    pred = pd.read_parquet("preds/baseline_quantiles.parquet")
    feat = pd.read_parquet("data/proc/features.parquet")
    taus = np.arange(0.05,0.96,0.05)
    qcols = [f"q{int(t*100):02d}" for t in taus]
    mask = pred[qcols].notna().all(axis=1)
    L = pinball(feat.loc[mask,"ret_1d"].values, pred.loc[mask,qcols].values, taus)
    Path("reports").mkdir(parents=True, exist_ok=True)
    out = Path("reports/report.txt")
    out.write_text(f"pinball_avg={L:.6f}\n")
    print(str(out))

if __name__ == "__main__":
    main()
