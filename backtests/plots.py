from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

TAUS = np.arange(0.05,0.96,0.05)
QCOLS = [f"q{int(t*100):02d}" for t in TAUS]

def pit_values(y, qs):
    y = np.asarray(y).reshape(-1,1)
    return (y <= qs).mean(axis=1)

def main():
    feat = pd.read_parquet("data/proc/features.parquet")
    y = feat["ret_1d"].to_numpy()
    Path("reports").mkdir(parents=True, exist_ok=True)
    for p in Path("preds").glob("*.parquet"):
        df = pd.read_parquet(p).merge(feat[["date","ret_1d"]], on="date", how="inner")
        qs = df[QCOLS].to_numpy()
        pit = pit_values(df["ret_1d"].to_numpy(), qs)
        plt.figure()
        plt.hist(pit, bins=20, density=True)
        plt.title(f"PIT: {p.stem}")
        out = Path("reports")/f"pit_{p.stem}.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(out)
if __name__ == "__main__":
    main()
