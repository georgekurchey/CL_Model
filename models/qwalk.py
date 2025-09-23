from __future__ import annotations
import numpy as np

class QuantumWalkMC:
    def __init__(self, ret, rv_win=20, mix=0.3, df=5, seed=0):
        self.r = np.asarray(ret, float)
        self.rv_win = int(rv_win)
        self.mix = float(mix)
        self.df = float(df)
        self.rng = np.random.default_rng(seed)
        self.sig = None

    def fit(self):
        r = self.r
        w = np.ones(self.rv_win)/self.rv_win
        rv = np.sqrt(np.convolve(r**2, w, mode="same") + 1e-12)
        rv[np.isnan(rv)] = np.nanmean(rv)
        self.sig = rv
        return self

    def _sample_steps(self, n_paths):
        n = len(self.r)
        sig = self.sig.reshape(-1,1)
        u = self.rng.uniform(size=(n, n_paths))
        walk = np.sign(u - 0.5) * sig
        shock = self.rng.standard_t(self.df, size=(n, n_paths)) * sig
        return (1.0 - self.mix)*walk + self.mix*shock

    def quantiles(self, taus, n_paths=4000):
        X = self._sample_steps(n_paths=n_paths)
        qs = np.quantile(X, np.asarray(taus), axis=1).T
        return qs
