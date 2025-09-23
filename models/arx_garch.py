from __future__ import annotations
import numpy as np, pandas as pd
from arch.univariate import ConstantMean, GARCH, Normal
from scipy.stats import norm

def fit_predict(df: pd.DataFrame, y_col='ret_1d', x_cols=(), taus=tuple(np.arange(0.05,0.96,0.05)), win=252):
    r = df[y_col].astype(float).to_numpy()
    X = df[list(x_cols)].astype(float).to_numpy() if x_cols else None
    mu = np.zeros_like(r)
    sig = np.full_like(r, np.nan, dtype=float)
    for t in range(len(r)):
        if t < win:
            continue
        y_tr = r[t-win:t]
        cm = ConstantMean(y_tr)
        cm.volatility = GARCH(1,0,1)
        cm.distribution = Normal()
        try:
            res = cm.fit(disp='off')
            f = res.forecast(horizon=1, reindex=False)
            mu[t] = float(res.params.get('mu', 0.0))
            sig[t] = float(np.sqrt(f.variance.values[-1,0]))
        except Exception:
            mu[t] = 0.0
            sig[t] = np.nan
    z = norm.ppf(np.asarray(taus))
    Q = (mu[:,None] + np.nan_to_num(sig)[:,None]*z[None,:])
    out = pd.DataFrame({'date': df['date']})
    for i,tau in enumerate(taus):
        out[f"q{int(tau*100):02d}"] = Q[:,i]
    return out
