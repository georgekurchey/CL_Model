from pathlib import Path
import pandas as pd
from features.build_features import build_features
from backtests.walkforward import walkforward

def test_build_features(tmp_path, monkeypatch):
    out = build_features()
    assert Path(out).exists()
    df = pd.read_parquet(out)
    assert {"ret_1d","curve_level","curve_slope","curve_curv"}.issubset(df.columns)

def test_walkforward(tmp_path):
    out = build_features()
    assert Path(out).exists()
    walkforward("config/pipeline.yaml")
    assert Path("preds/baseline_quantiles.parquet").exists()
