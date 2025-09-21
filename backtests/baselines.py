from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from models.arx_garch import fit_predict as arxgarch
from models.quantile_baseline import fit_predict as qreg

TAUS = tuple(np.arange(0.05,0.96,0.05))

def feature_cols(df: pd.DataFrame):
    cand = ["usd_z","real_rate_10y","term_slope","curve_slope","eia_us_crude_surprise","eia_cushing_surprise","eia_runs_surprise"]
    return [c for c in cand if c in df.columns]

def run(features_path='data/proc/features.parquet'):
    df = pd.read_parquet(features_path)
    xcols = feature_cols(df)
    Path('preds').mkdir(parents=True, exist_ok=True)
    q1 = qreg(df, y_col='ret_1d', x_cols=xcols, taus=TAUS, win=252)
    q1.to_parquet('preds/quantile_reg.parquet', index=False)
    a1 = arxgarch(df, y_col='ret_1d', x_cols=xcols, taus=TAUS, win=252)
    a1.to_parquet('preds/arx_garch.parquet', index=False)
    return {'quantile_reg':'preds/quantile_reg.parquet','arx_garch':'preds/arx_garch.parquet'}

if __name__ == '__main__':
    print(run())
