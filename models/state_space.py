from __future__ import annotations
import numpy as np, pandas as pd
from .common import to_quantiles

def _add_const(X):
    X = np.asarray(X, float)
    c = np.ones((len(X),1))
    return np.hstack([c, X])

def _ols(X, y, ridge=1e-6):
    XtX = X.T @ X + ridge*np.eye(X.shape[1])
    Xty = X.T @ y
    return np.linalg.solve(XtX, Xty)

class SVJump1D:
    def __init__(self, ret, exog=None, win=252, seed=0):
        self.r = np.asarray(ret, float)
        self.X = None if exog is None else np.asarray(exog, float)
        self.win = int(win)
        self.rng = np.random.default_rng(seed)
        self.params = None

    def fit(self):
        r = self.r
        n = len(r)
        mu = np.zeros(n)
        sig = np.zeros(n)
        if self.X is None:
            roll_mu = pd.Series(r).rolling(self.win).mean().fillna(0.0).to_numpy()
            roll_sd = pd.Series(r).rolling(self.win).std(ddof=0).fillna(pd.Series(r).std(ddof=0)).to_numpy()
            med = float(np.nanmedian(r))
            mad = float(np.nanmedian(np.abs(r - med))) or 1e-8
            jump = np.where(np.abs(r - med) > 3.0*mad, r, 0.0)
            mu[:] = roll_mu
            sig[:] = np.sqrt(roll_sd**2 + 0.5*jump**2)
        else:
            X = self.X
            for t in range(n):
                if t < self.win:
                    mu[t] = 0.0
                    sig[t] = np.std(r[:t+1]) if t>10 else np.std(r[:max(2,t+1)])
                    continue
                Xt = _add_const(X[t-self.win:t])
                yt = r[t-self.win:t]
                b = _ols(Xt, yt)
                mu[t] = float(np.r_[1.0, X[t]].dot(b))
                res = yt - Xt @ b
                lv = np.log(np.abs(res)+1e-8)
                b2 = _ols(Xt, lv)
                lsig = float(np.r_[1.0, X[t]].dot(b2))
                sig[t] = float(np.exp(lsig))
        self.params = {"mu": mu, "sigma": sig}
        return self

    def predict_quantiles(self, taus):
        return to_quantiles(self.params["mu"], self.params["sigma"], taus)
