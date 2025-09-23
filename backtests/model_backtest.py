from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from models.state_space import SVJump1D

TAUS = np.arange(0.05,0.96,0.05)

def ss_factors(df):
    cols = [c for c in df.columns if c.startswith("CL_M")]
    X = df[cols].to_numpy(float)
    x = np.log(np.clip(X, 1e-9, None))
    lvl = x.mean(axis=1)
    dev = x - lvl[:,None]
    slope = dev[:,0] - dev[:,-1]
    mid = dev[:,len(cols)//2]
    curv = dev[:,0] - 2*mid + dev[:,-1]
    return pd.DataFrame({"ss_level":lvl, "ss_slope":slope, "ss_curv":curv})

BASE_FEATS = [
    "usd_z","real_rate_10y","term_slope",
    "eia_us_crude_surprise","eia_cushing_surprise","eia_runs_surprise",
    "roll_window_dummy","eia_day_dummy"
]

def run(features_path='data/proc/features.parquet', out_path='preds/model_svjump.parquet', ridge=1e-4):
    df = pd.read_parquet(features_path)
    ss = ss_factors(df)
    use = [c for c in BASE_FEATS if c in df.columns]
    X = pd.concat([df[use], ss], axis=1).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    r = df["ret_1d"].astype(float).fillna(0.0).to_numpy()
    m = SVJump1D(r, exog=X.to_numpy(float), win=252, ridge=float(ridge)).fit()
    Q = m.predict_quantiles(TAUS)
    out = pd.DataFrame({"date": df["date"]})
    for i,t in enumerate(TAUS):
        out[f"q{int(t*100):02d}"] = Q[:,i]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    Path("reports").mkdir(parents=True, exist_ok=True)
    names = use + ["ss_level","ss_slope","ss_curv"]
    m.dump_params("reports/model_params.json", feature_names=names)
    return out_path

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-r","--ridge", type=float, default=1e-4)
    args = ap.parse_args()
    print(run(ridge=args.ridge))
