import pandas as pd
from features.build_features import build_features

def test_eia_surprises_present():
    out = build_features()
    df = pd.read_parquet(out)
    cols = {"eia_us_crude_surprise","eia_cushing_surprise","eia_runs_surprise"}
    assert cols.issubset(df.columns)
