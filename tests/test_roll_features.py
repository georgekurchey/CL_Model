from pathlib import Path
import pandas as pd
from features.build_features import build_features

def test_roll_window_present():
    out = build_features()
    assert Path(out).exists()
    df = pd.read_parquet(out)
    assert {"roll_flag","roll_window_dummy"}.issubset(df.columns)
    # sanity: at least one roll per calendar month on business days
    assert df["roll_flag"].sum() >= max(1, df["date"].dt.to_period("M").nunique() - 1)
