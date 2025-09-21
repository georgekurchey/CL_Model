from __future__ import annotations
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BMonthEnd, BusinessDay

def last_trading_day(date: pd.Timestamp) -> pd.Timestamp:
    # Approximate: last business day of the month
    d = pd.Timestamp(date).normalize()
    return (d + BMonthEnd(0))

def force_roll_deadline(dates: pd.DatetimeIndex, deadline_business_days: int = 5) -> pd.Series:
    """
    Return a boolean Series (index=dates) where True indicates the 'roll day'
    forced by the deadline rule: roll on (last_trading_day - deadline_bd).
    """
    dates = pd.DatetimeIndex(dates)
    # Compute month key for each date
    mkey = dates.to_period("M")
    # For each unique month in the sample, compute the forced roll date
    roll_map = {}
    for m in mkey.unique():
        month_start = pd.Timestamp(m.start_time)
        ltd = last_trading_day(month_start)
        roll_day = ltd - BusinessDay(deadline_business_days)
        roll_map[m] = roll_day.normalize()
    flags = pd.Series(False, index=dates)
    # Mark True where date == forced roll date
    for i, d in enumerate(dates):
        if d.normalize() == roll_map[mkey[i]]:
            flags.iloc[i] = True
    flags.name = "roll_flag"
    return flags

def roll_window_dummy(dates: pd.DatetimeIndex, roll_flags: pd.Series, window: int = 7) -> pd.Series:
    """
    ±window business days around each roll_flag True day.
    """
    dates = pd.DatetimeIndex(dates)
    rf_idx = np.flatnonzero(roll_flags.to_numpy())
    mask = np.zeros(len(dates), dtype=bool)
    for i in rf_idx:
        lo = max(0, i - window)
        hi = min(len(dates), i + window + 1)
        mask[lo:hi] = True
    return pd.Series(mask, index=dates, name="roll_window_dummy")
