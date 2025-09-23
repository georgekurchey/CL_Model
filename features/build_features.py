from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from .curve_pca import curve_pca_k3
from .roll import force_roll_deadline, roll_window_dummy
from .eia_features import build_eia_surprises

PROC = Path("data/proc"); PROC.mkdir(parents=True, exist_ok=True)

def _load_chris() -> pd.DataFrame:
    raw = Path("data/raw/chris")
    frames = []
    for depth in range(1, 13):
        f = raw / f"CL{depth}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, parse_dates=["date"])
        if "settle" in df.columns:
            df = df.rename(columns={"settle": f"CL_M{depth}"})
        frames.append(df[["date", f"CL_M{depth}"]])
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for k in frames[1:]:
        out = out.merge(k, on="date", how="outer")
    return out.sort_values("date")

def _fred_block() -> pd.DataFrame:
    raw = Path("data/raw/fred")
    parts = []
    for s in ["DTWEXBGS","DGS10","T10YIE"]:
        f = raw / f"{s}.csv"
        if f.exists():
            parts.append(pd.read_csv(f, parse_dates=["date"]))
    if not parts:
        return pd.DataFrame()
    df = parts[0]
    for k in parts[1:]:
        df = df.merge(k, on="date", how="outer")
    return df.sort_values("date")

def _zscore(x, win=252):
    r = x.rolling(win, min_periods=max(30,win//5))
    return (x - r.mean())/r.std(ddof=0)

def build_features(config_path: str = "config/pipeline.yaml") -> Path:
    cfg = yaml.safe_load(Path(config_path).read_text())
    curve = _load_chris()
    if curve.empty:
        dates = pd.bdate_range("2020-01-01", periods=90)
        toy = {f"CL_M{i}": 70 + 0.03*np.arange(len(dates)) + 0.15*i for i in range(1,13)}
        curve = pd.DataFrame(toy, index=dates).reset_index().rename(columns={"index":"date"})
    fred = _fred_block()
    df = curve.merge(fred, on="date", how="left")

    df["ret_1d"] = np.log(df["CL_M1"]).diff().fillna(0.0)

    curve_only = df.filter(like="CL_M")
    pca = curve_pca_k3(curve_only.set_index(df["date"]))
    pca = pca.reset_index().rename(columns={"index":"date"})
    out = df.merge(pca, on="date", how="left").sort_values("date").reset_index(drop=True)

    dl_bd = int(cfg.get("roll_rule", {}).get("deadline_business_days", 5))
    dates = pd.DatetimeIndex(out["date"])
    rf = force_roll_deadline(dates, deadline_business_days=dl_bd)
    out["roll_flag"] = rf.values
    out["roll_window_dummy"] = roll_window_dummy(dates, rf, window=7).values

    eia_daily = build_eia_surprises(out["date"])
    out = out.merge(eia_daily, on="date", how="left")

    try:
        cvol = pd.read_parquet("data/raw/cvol/wti_cvol.parquet")
        out = out.merge(cvol, on="date", how="left")
        out["cvol_1m_z"] = _zscore(out["cvol_1m"])
        out["skew_25d_z"] = _zscore(out["skew_25d"])
        out["term_premium_z"] = _zscore(out["term_premium"])
        out["iv_rv_gap"] = out["cvol_1m"] - out["ret_1d"].rolling(20, min_periods=10).std(ddof=0)*252**0.5
    except Exception:
        pass
    try:
        cot = pd.read_parquet("data/raw/cftc/managed_money.parquet")
        out = out.merge(cot, on="date", how="left")
        if "open_interest" in out and "mm_net" in out:
            out["mm_net_norm"] = out["mm_net"] / out["open_interest"].replace(0, pd.NA)
        out["mm_net_chg_4w"] = out.get("mm_net", pd.Series(index=out.index)).diff(4)
    except Exception:
        pass

    if {"DTWEXBGS","DGS10","T10YIE"}.issubset(out.columns):
        out["usd_z"] = _zscore(out["DTWEXBGS"])
        out["real_rate_10y"] = out["DGS10"] - out["T10YIE"]
        out["term_slope"] = out["DGS10"] - out["T10YIE"]
        out["real_rate_10y_z"] = _zscore(out["real_rate_10y"])
        out["term_slope_z"] = _zscore(out["term_slope"])

    out_file = PROC / "features.parquet"
    out.to_parquet(out_file, index=False)
    return out_file

def main(config: str) -> None:
    out = build_features(config)
    print(str(out))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pipeline.yaml")
    args = ap.parse_args()
    main(args.config)
