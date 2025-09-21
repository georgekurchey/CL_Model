from __future__ import annotations
import numpy as np
import pandas as pd

def pinball(y, qhat, taus):
    y = np.asarray(y)
    L = 0.0
    for i,tau in enumerate(taus):
        e = y - qhat[:,i]
        L += np.mean(np.maximum(tau*e, (tau-1)*e))
    return L/len(taus)

def pit(y, qhat, taus):
    y = np.asarray(y)
    qs = qhat
    m = (y.reshape(-1,1) <= qs).mean(axis=0)
    return m
