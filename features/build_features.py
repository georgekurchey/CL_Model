from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from .curve_pca import curve_pca_k3
from .roll import force_roll_deadline, roll_window_dummy

PROC = Path("data/proc"); PROC.mkdir(parents=True, exist_ok=True)

def _load_cme_continuous() -> pd.DataFrame:
    """
    Reads data/raw/chris/CL1.csv ... CL12.csv (Nasdaq Data Link CHRIS continuous futures)
    and maps them to columns CL_M1..CL_M12 by date.
    """
    raw = Path("data/raw/chris")
    frames = []
    for depth in range(1, 13):
        f = raw / f"CL{depth}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, parse_dates=["date"])
        df = df.rename(columns={"settle": f"CL_M{depth}"})
        frames.append(df[["date", f"CL_M{depth}"]])
    if not frames:
        raise SystemExit("No CME continuous files found under data/raw/chris (run make ingest)")
    out = frames[0]
    for k in frames[1:]:
        out = out.merge(k, on="date", how="outer")
    return out.sort_values("date")

def _fred_block() -> pd.DataFrame:
    raw = Path("data/raw/fred")
    out = []
    for s in ["DTWEXBGS","DGS10","T10YIE"]:
        f = raw / f"{s}.csv"
        if f.exists():
            out.append(pd.read_csv(f, parse_dates=["date"]))
    if not out:
        return pd.DataFrame()
    df = out[0]
    for k in out[1:]:
        df = df.merge(k, on="date", how="outer")
    return df.sort_values("date")

def build_features(config_path: str = "config/pipeline.yaml") -> Path:
    cfg = yaml.safe_load(Path(config_path).read_text())
    curve = _load_cme_continuous()                        # ← real settlements now
    fred = _fred_block()
    df = curve.merge(fred, on="date", how="left")

    # Targets (primary)
    df["ret_1d"] = np.log(df["CL_M1"]).diff().fillna(0.0)

    # Curve PCA (on M1..M12)
    curve_only = df.filter(like="CL_M").copy()
    pca = curve_pca_k3(curve_only.set_index(df["date"]))
    pca = pca.reset_index().rename(columns={"index": "date"})
    out = df.merge(pca, on="date", how="left").sort_values("date").reset_index(drop=True)

    # Roll flags & roll-window dummy (deadline rule from config)
    dl_bd = int(cfg.get("roll_rule", {}).get("deadline_business_days", 5))
    dates = pd.DatetimeIndex(out["date"])
    rf = force_roll_deadline(dates, deadline_business_days=dl_bd)
    out["roll_flag"] = rf.values
    out["roll_window_dummy"] = roll_window_dummy(dates, rf, window=7).values

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
