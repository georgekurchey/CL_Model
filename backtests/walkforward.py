from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import yaml
from .scoring import save_dummy_report

PRED = Path("preds"); PRED.mkdir(exist_ok=True)

def _load_features() -> pd.DataFrame:
    f = Path("data/proc/features.parquet")
    if not f.exists():
        raise SystemExit("missing data/proc/features.parquet (run features step)")
    return pd.read_parquet(f)

def walkforward(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    df = _load_features()
    preds = df[["date","ret_1d"]].copy()
    preds["q05"] = preds["ret_1d"].rolling(50, min_periods=20).quantile(0.05).fillna(0)
    preds["q50"] = preds["ret_1d"].rolling(50, min_periods=20).quantile(0.50).fillna(0)
    preds["q95"] = preds["ret_1d"].rolling(50, min_periods=20).quantile(0.95).fillna(0)
    out = PRED / "baseline_quantiles.parquet"
    preds.to_parquet(out, index=False)
    save_dummy_report(preds, Path(cfg["reports_dir"]))
    print(str(out))

def main(config: str) -> None:
    walkforward(config)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pipeline.yaml")
    args = ap.parse_args()
    main(args.config)
