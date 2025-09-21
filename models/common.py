from __future__ import annotations
import numpy as np

def to_quantiles(mu, sigma, taus):
    from scipy.stats import norm
    mu = np.asarray(mu).reshape(-1)
    sigma = np.asarray(sigma).reshape(-1)
    z = norm.ppf(np.asarray(taus))
    return mu[:,None] + sigma[:,None]*z[None,:]
