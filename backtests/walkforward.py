from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

TAUS = np.arange(0.05, 0.96, 0.05)

def rolling_quantiles(ret, win=252):
    q = []
    r = ret.values
    for i in range(len(r)):
        if i < win:
            q.append([0.0]*len(TAUS))
        else:
            q.append(np.quantile(r[i-win:i], TAUS).tolist())
    return np.array(q)

def run_backtest(features_path="data/proc/features.parquet", out_path="preds/baseline_quantiles.parquet"):
    df = pd.read_parquet(features_path)
    ret = df["ret_1d"].astype(float).fillna(0.0)
    Q = rolling_quantiles(ret, win=252)
    out = pd.DataFrame({"date": df["date"]})
    for i,t in enumerate(TAUS):
        out[f"q{int(t*100):02d}"] = Q[:,i]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out_path

def walkforward(config_path: str = "config/pipeline.yaml"):
    return run_backtest()

if __name__ == "__main__":
    print(run_backtest())
