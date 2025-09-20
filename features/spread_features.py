from pathlib import Path
import numpy as np
import pandas as pd

def _roll_z(x: pd.Series, win: int = 252) -> pd.Series:
    m = x.rolling(win, min_periods=max(20, win//4)).mean()
    s = x.rolling(win, min_periods=max(20, win//4)).std()
    return (x - m) / s

def merge_spread_features(df_features: pd.DataFrame,
                          brent_csv: str,
                          wti_price_col: str = "px_cm1",
                          lookback: int = 252) -> pd.DataFrame:
    df = df_features.copy()
    df["date"] = pd.to_datetime(df["date"])
    b = pd.read_csv(brent_csv, parse_dates=["date"])
    b = b.sort_values("date")
    # forward-fill brent to align with business days
    b = b.set_index("date").asfreq("B").ffill().reset_index()
    if wti_price_col not in df.columns:
        raise ValueError(f"{wti_price_col} not in features; add roll utils first or choose another column")
    merged = df.merge(b[["date","brent_close"]], on="date", how="left")
    # spread and z-score
    merged["wti_brent_spread"] = merged[wti_price_col] - merged["brent_close"]
    merged["wti_brent_z"] = _roll_z(merged["wti_brent_spread"], lookback)
    # 1d Brent return
    merged["brent_ret_1d"] = merged["brent_close"].pct_change()
    return merged
