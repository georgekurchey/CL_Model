import pandas as pd
import numpy as np
from features.roll_utils import detect_roll_flags, compute_spliced_returns, synth_constant_m1

def _toy_strip():
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    m1 = [70.0, 70.5, 71.0, 72.0, 72.3, 72.8]
    m2 = [71.0, 71.2, 72.0, 72.5, 73.0, 73.2]
    wide = pd.DataFrame({"date": dates, "m1": m1, "m2": m2})
    wide.loc[3, "m1"] = wide.loc[2, "m2"]  # force roll detection at t=3
    return wide

def test_detect_roll():
    dfw = _toy_strip()
    rf = detect_roll_flags(dfw, eps=0.005)
    assert rf.sum() == 1 and rf.index[rf.values][0] == 3

def test_spliced_return():
    dfw = _toy_strip(); rf = detect_roll_flags(dfw, eps=0.01)
    rs = compute_spliced_returns(dfw, rf)
    assert abs(rs.iloc[3]) < 1e-9

def test_cm1_chain():
    dfw = _toy_strip(); rf = detect_roll_flags(dfw, eps=0.01)
    rs = compute_spliced_returns(dfw, rf)
    px = synth_constant_m1(dfw, rs)
    assert np.isfinite(px.dropna()).all()
