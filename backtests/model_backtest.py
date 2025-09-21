from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from models.state_space import SVJump1D

TAUS = np.arange(0.05,0.96,0.05)

FEAT_COLS = [
    "usd_z","real_rate_10y","term_slope",
    "eia_us_crude_surprise","eia_cushing_surprise","eia_runs_surprise",
    "roll_window_dummy"
]

def run(features_path='data/proc/features.parquet', out_path='preds/model_svjump.parquet'):
    df = pd.read_parquet(features_path)
    r = df["ret_1d"].astype(float).fillna(0.0).to_numpy()
    cols = [c for c in FEAT_COLS if c in df.columns]
    X = df[cols].astype(float).to_numpy() if cols else None
    m = SVJump1D(r, exog=X, win=252).fit()
    Q = m.predict_quantiles(TAUS)
    out = pd.DataFrame({"date": df["date"]})
    for i,t in enumerate(TAUS):
        out[f"q{int(t*100):02d}"] = Q[:,i]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out_path

if __name__ == "__main__":
    print(run())
