from __future__ import annotations
import numpy as np
from typing import Dict, List

def _nonconformity(y: np.ndarray, q: np.ndarray, side: str) -> np.ndarray:
    if side == "lower":
        return np.maximum(q - y, 0.0)
    elif side == "upper":
        return np.maximum(y - q, 0.0)
    raise ValueError("side must be 'lower' or 'upper'")

def _finite_sample_quantile(scores: np.ndarray, u: float) -> float:
    s = np.asarray(scores, float)
    s = s[np.isfinite(s)]
    n = s.shape[0]
    if n == 0:
        return 0.0
    k = int(np.ceil(n * u))
    k = min(max(k, 1), n)
    return float(np.partition(s, k-1)[k-1])

def conformal_offsets(
    y_cal: np.ndarray,
    q_cal: np.ndarray,
    taus: List[float],
    targets: List[float]
) -> Dict[float, float]:
    taus = [float(t) for t in taus]
    t2idx = {float(t): i for i,t in enumerate(taus)}
    offs: Dict[float,float] = {}
    for t in targets:
        if t not in t2idx:
            raise ValueError(f"target {t} not in taus; include it in the quantile grid")
        qi = q_cal[:, t2idx[t]]
        if t < 0.5:
            scores = _nonconformity(y_cal, qi, "lower")
            u = 1.0 - t
            offs[t] = _finite_sample_quantile(scores, u)
        else:
            scores = _nonconformity(y_cal, qi, "upper")
            u = t
            offs[t] = _finite_sample_quantile(scores, u)
    return offs

def apply_tail_offsets(q_pred: np.ndarray, taus: List[float], offsets: Dict[float,float], tail_only: bool=True) -> np.ndarray:
    q = np.array(q_pred, float, copy=True)
    taus = [float(t) for t in taus]
    t2idx = {float(t): i for i,t in enumerate(taus)}
    for t,c in offsets.items():
        if t not in t2idx:
            continue
        idx = t2idx[t]
        if t < 0.5:
            if tail_only:
                q[:, idx] = q[:, idx] + c
            else:
                for j, tt in enumerate(taus):
                    if tt <= t:
                        q[:, j] = q[:, j] + c
        else:
            if tail_only:
                q[:, idx] = q[:, idx] - c
            else:
                for j, tt in enumerate(taus):
                    if tt >= t:
                        q[:, j] = q[:, j] - c
    q = np.maximum.accumulate(q, axis=1)
    return q
