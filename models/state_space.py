from __future__ import annotations
import json
import numpy as np, pandas as pd
from .common import to_quantiles

def _add_const(X):
    X = np.asarray(X, float)
    return np.column_stack([np.ones(len(X)), X])

def _ols(X, y, ridge=1e-6):
    XtX = X.T @ X + ridge*np.eye(X.shape[1])
    Xty = X.T @ y
    return np.linalg.solve(XtX, Xty)

class SVJump1D:
    def __init__(self, ret, exog=None, win=252, ridge=1e-4, seed=0):
        self.r = np.asarray(ret, float)
        self.X = None if exog is None else np.asarray(exog, float)
        self.win = int(win)
        self.ridge = float(ridge)
        self.params = None
        self.coefs_ = {}
        self.rng = np.random.default_rng(seed)

    def fit(self):
        r = self.r
        n = len(r)
        mu = np.zeros(n)
        sig = np.zeros(n)
        self.coefs_ = {"beta_mu": None, "beta_lv": None}

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
            p = X.shape[1] + 1
            beta_mu = np.zeros(p)
            beta_lv = np.zeros(p)
            for t in range(n):
                if t < self.win:
                    mu[t] = 0.0
                    sig[t] = np.std(r[:max(2,t+1)])
                    continue
                Xt = _add_const(X[t-self.win:t])
                yt = r[t-self.win:t]
                b = _ols(Xt, yt, ridge=self.ridge)
                mu[t] = float(np.r_[1.0, X[t]].dot(b))
                res = yt - Xt @ b
                lv = np.log(np.abs(res) + 1e-8)
                b2 = _ols(Xt, lv, ridge=self.ridge)
                lsig = float(np.r_[1.0, X[t]].dot(b2))
                sig[t] = float(np.exp(lsig))
                beta_mu = 0.98*beta_mu + 0.02*b
                beta_lv = 0.98*beta_lv + 0.02*b2
            self.coefs_["beta_mu"] = beta_mu.tolist()
            self.coefs_["beta_lv"] = beta_lv.tolist()

        self.params = {"mu": mu, "sigma": sig}
        return self

    def predict_quantiles(self, taus):
        return to_quantiles(self.params["mu"], self.params["sigma"], taus)

    def dump_params(self, path:str, feature_names=None):
        d = {
            "ridge": self.ridge,
            "win": self.win,
            "beta_mu": self.coefs_.get("beta_mu"),
            "beta_lv": self.coefs_.get("beta_lv"),
            "feature_names": feature_names
        }
        with open(path, "w") as f:
            json.dump(d, f, indent=2)
