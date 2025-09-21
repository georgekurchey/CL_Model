from __future__ import annotations
import numpy as np, pandas as pd
from .common import to_quantiles

class SVJump1D:
    def __init__(self, ret, seed=0):
        self.r = np.asarray(ret, float)
        self.rng = np.random.default_rng(seed)
        self.params_ = None

    def fit(self, win=252):
        r = pd.Series(self.r)
        mu = r.rolling(win).mean().fillna(0.0).to_numpy()
        vol = r.rolling(win).std(ddof=0).fillna(r.std(ddof=0)).to_numpy()
        med = float(np.nanmedian(r))
        mad = float(np.nanmedian(np.abs(r - med))) or 1e-8
        jump = np.where(np.abs(r - med) > 3.0*mad, r.to_numpy(), 0.0)
        sigma = np.sqrt(vol**2 + 0.5*jump**2)
        self.params = {"mu": mu, "sigma": sigma}
        return self

    def predict_quantiles(self, taus):
        return to_quantiles(self.params["mu"], self.params["sigma"], taus)
