from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.linear_model import QuantileRegressor

def fit_predict(df: pd.DataFrame, y_col='ret_1d', x_cols=(), taus=tuple(np.arange(0.05,0.96,0.05)), win=252, alpha=0.0001):
    y = df[y_col].astype(float).to_numpy()
    X = df[list(x_cols)].astype(float).to_numpy() if x_cols else None
    if X is None:
        X = np.ones((len(y),1))
    Qmat = np.zeros((len(y), len(taus)))
    for t in range(len(y)):
        if t < win:
            continue
        Xt = X[t-win:t]
        yt = y[t-win:t]
        for j,tau in enumerate(taus):
            qr = QuantileRegressor(quantile=float(tau), alpha=alpha, fit_intercept=True)
            try:
                qr.fit(Xt, yt)
                Qmat[t,j] = float(qr.predict(X[t].reshape(1,-1)))
            except Exception:
                Qmat[t,j] = 0.0
    out = pd.DataFrame({'date': df['date']})
    for j,tau in enumerate(taus):
        out[f"q{int(tau*100):02d}"] = Qmat[:,j]
    return out
