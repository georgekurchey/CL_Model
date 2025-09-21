import pandas as pd
from features.build_features import build_features

def test_macro_features_present():
    out = build_features()
    df = pd.read_parquet(out)
    cols = {"usd_z","real_rate_10y","term_slope"}
    assert cols.issubset(df.columns)
