from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

EIA_DIR = Path("data/raw/eia")

def _read_any(cands, value_cols=("value","VALUE","val")):
    for c in cands:
        p = EIA_DIR / c
        if p.exists():
            df = pd.read_csv(p, parse_dates=["date"])
            if "date" not in df.columns:
                if "Date" in df.columns: df = df.rename(columns={"Date":"date"})
            if not set([*value_cols]).intersection(df.columns):
                for k in df.columns:
                    if k not in ("date",):
                        df = df.rename(columns={k:"value"})
                        break
            if "value" not in df.columns:
                for k in value_cols:
                    if k in df.columns:
                        df = df.rename(columns={k:"value"})
                        break
            return df[["date","value"]].sort_values("date").dropna()
    return pd.DataFrame(columns=["date","value"])

def load_eia_weeklies():
    us_crude = _read_any(["PET.WCESTUS1.W.csv","weekly_crude_stocks.csv"])
    cushing = _read_any(["PET.WCSSTUS1.W.csv","weekly_cushing_stocks.csv","cushing_stocks.csv"])
    runs = _read_any(["PET.WCRRIUS2.W.csv","weekly_refinery_inputs.csv","refinery_inputs.csv"])
    return {"us_crude": us_crude, "cushing": cushing, "runs": runs}

def _surprise(df, win=260):
    s = df.copy()
    s = s.dropna().sort_values("date")
    s["seasonal"] = s["value"].rolling(win, min_periods=26).mean()
    s["resid"] = s["value"] - s["seasonal"]
    z = s["resid"].shift(1)
    y = s["resid"]
    mask = z.notna() & y.notna()
    if mask.sum() >= 20:
        phi = float(np.cov(z[mask], y[mask], ddof=0)[0,1] / np.var(z[mask], ddof=0))
    else:
        phi = 0.0
    s["pred"] = phi * s["resid"].shift(1)
    s["surprise"] = (s["resid"] - s["pred"]).fillna(0.0)
    s = s[["date","value","surprise"]]
    s = s.rename(columns={"value":"actual"})
    return s

def map_weekly_to_daily_surprise(wdf, daily_index):
    x = wdf.copy()
    x["effective_date"] = (x["date"] + BDay(1)).dt.normalize()
    d = pd.DataFrame({"date": pd.DatetimeIndex(daily_index).normalize()}).drop_duplicates()
    d = d.merge(x[["effective_date","surprise"]], left_on="date", right_on="effective_date", how="left").drop(columns=["effective_date"])
    d["surprise"] = d["surprise"].ffill().fillna(0.0)
    return d

def build_eia_surprises(daily_index):
    wk = load_eia_weeklies()
    out = {}
    for k, df in wk.items():
        if not df.empty:
            out[k] = _surprise(df)
    daily = pd.DataFrame({"date": pd.DatetimeIndex(daily_index)})
    if "us_crude" in out:
        daily["eia_us_crude_surprise"] = map_weekly_to_daily_surprise(out["us_crude"], daily["date"])["surprise"]
    else:
        daily["eia_us_crude_surprise"] = 0.0
    if "cushing" in out:
        daily["eia_cushing_surprise"] = map_weekly_to_daily_surprise(out["cushing"], daily["date"])["surprise"]
    else:
        daily["eia_cushing_surprise"] = 0.0
    if "runs" in out:
        daily["eia_runs_surprise"] = map_weekly_to_daily_surprise(out["runs"], daily["date"])["surprise"]
    else:
        daily["eia_runs_surprise"] = 0.0
    return daily
