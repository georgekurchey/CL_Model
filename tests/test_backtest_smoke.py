from backtests.walkforward import run_backtest
import pandas as pd

def test_backtest_outputs():
    p = run_backtest()
    df = pd.read_parquet(p)
    assert set(df.columns).issuperset({"date","q05","q50","q95"})
