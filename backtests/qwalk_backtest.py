from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from models.qwalk import QuantumWalkMC

TAUS = np.arange(0.05,0.96,0.05)

def run(features_path='data/proc/features.parquet', out_path='preds/model_qwalk.parquet',
        mix=0.45, dof=4, paths=8000):
    feats = pd.read_parquet(features_path)
    r = feats['ret_1d'].astype(float).fillna(0.0).to_numpy()
    m = QuantumWalkMC(r, mix=float(mix), df=float(dof)).fit()
    Q = m.quantiles(TAUS, n_paths=int(paths))
    out = pd.DataFrame({'date': feats['date']})
    for i,t in enumerate(TAUS):
        out[f"q{int(t*100):02d}"] = Q[:,i]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out_path

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--mix', type=float, default=0.45)
    ap.add_argument('--df',  dest='dof', type=float, default=4)
    ap.add_argument('--paths', type=int, default=8000)
    args = ap.parse_args()
    print(run(mix=args.mix, dof=args.dof, paths=args.paths))
