from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from models.qwalk import QuantumWalkMC

TAUS = np.arange(0.05,0.96,0.05)

def run(features_path='data/proc/features.parquet', out_path='preds/model_qwalk.parquet'):
    df = pd.read_parquet(features_path)
    r = df['ret_1d'].astype(float).fillna(0.0).to_numpy()
    m = QuantumWalkMC(r).fit()
    Q = m.quantiles(TAUS)
    out = pd.DataFrame({'date': df['date']})
    for i,t in enumerate(TAUS):
        out[f"q{int(t*100):02d}"] = Q[:,i]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out_path

if __name__ == '__main__':
    print(run())
