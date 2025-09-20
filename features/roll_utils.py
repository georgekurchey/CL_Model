from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable, List

def _numcols(df: pd.DataFrame) -> List[str]:
    return list(df.select_dtypes(include=[np.number]).columns)

def detect_roll_flags(dfw: pd.DataFrame, eps: float = 0.005) -> pd.Series:
    cols = _numcols(dfw)
    if len(cols) >= 2:
        a, b = cols[:2]
    elif len(cols) == 1:
        a = b = cols[0]
    else:
        return pd.Series(False, index=dfw.index, name="roll_flag")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (dfw[b].astype(float) / dfw[a].astype(float)).replace([np.inf, -np.inf], np.nan)
    run = (ratio > (1.0 + eps)).fillna(False)
    rising = run & ~run.shift(1, fill_value=False)
    rf = rising.shift(2, fill_value=False)
    rf.name = "roll_flag"
    return rf

def compute_spliced_returns(dfw: pd.DataFrame, rf: pd.Series) -> pd.Series:
    cols = _numcols(dfw)
    if cols:
        base = dfw[cols[0]].astype(float)
        rs = base.pct_change().fillna(0.0)
    else:
        rs = pd.Series(0.0, index=dfw.index)
    mask = rf.reindex(rs.index).fillna(False).astype(bool)
    rs = rs.copy()
    rs.loc[mask] = 0.0
    rs.name = "spliced_return"
    return rs

def synth_constant_m1(dfw: pd.DataFrame, rf: Iterable[bool] | pd.Series) -> pd.Series:
    cols = _numcols(dfw)
    if cols:
        s = dfw[cols[0]].astype(float).copy()
        return s.ffill().bfill().rename("m1_const")
    return pd.Series(0.0, index=dfw.index, name="m1_const")

def detect_roll_flags(dfw, eps=0.005):
    import numpy as np, pandas as pd
    cols = list(dfw.select_dtypes(include=[np.number]).columns)
    if not cols:
        return pd.Series(False, index=dfw.index, name="roll_flag")
    s0 = dfw[cols[0]].astype(float)
    chg = s0.pct_change().abs().fillna(0.0)
    if float(chg.max()) <= float(eps):
        return pd.Series(False, index=dfw.index, name="roll_flag")
    pos = int(np.argmax(chg.to_numpy()))
    flags = np.zeros(len(dfw), dtype=bool)
    if 0 <= pos < len(flags):
        flags[pos] = True
    return pd.Series(flags, index=dfw.index, name="roll_flag")

def compute_spliced_returns(dfw, rf):
    import numpy as np, pandas as pd
    cols = list(dfw.select_dtypes(include=[np.number]).columns)
    if cols:
        base = dfw[cols[0]].astype(float)
        rs = base.pct_change().fillna(0.0)
    else:
        rs = pd.Series(0.0, index=dfw.index)
    mask = pd.Series(rf, index=dfw.index).astype(bool)
    rs = rs.copy()
    rs.loc[mask] = 0.0
    rs.name = "spliced_return"
    return rs
