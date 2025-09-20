from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from .curve_pca import curve_pca_k3

PROC = Path("data/proc"); PROC.mkdir(parents=True, exist_ok=True)

def _toy_curve() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    data = {f"CL_M{i}": 70 + 0.05*np.arange(len(dates)) + 0.2*i for i in range(1, 13)}
    df = pd.DataFrame(data, index=dates).reset_index().rename(columns={"index":"date"})
    return df

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
    _ = yaml.safe_load(Path(config_path).read_text())
    curve = _toy_curve()
    fred = _fred_block()
    df = curve.merge(fred, on="date", how="left")
    df["ret_1d"] = np.log(df["CL_M1"]).diff().fillna(0.0)
    pca = curve_pca_k3(df.filter(like="CL_M").set_index(df["date"]))
    pca = pca.reset_index().rename(columns={"index": "date"})
    out = df.merge(pca, on="date", how="left").sort_values("date").reset_index(drop=True)
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
