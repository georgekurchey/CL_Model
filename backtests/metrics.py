from __future__ import annotations
import numpy as np

def crps_from_quantiles(y, qhat, taus):
    y = np.asarray(y).reshape(-1,1)
    qs = np.asarray(qhat)
    taus = np.asarray(taus).reshape(1,-1)
    Fhat = (y <= qs).astype(float)
    return float(np.mean(np.abs(Fhat - taus)))

def pit_values(y, qhat, taus):
    y = np.asarray(y).reshape(-1,1)
    qs = np.asarray(qhat)
    return (y <= qs).mean(axis=1)
