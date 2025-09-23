from pathlib import Path
import pandas as pd
import numpy as np
from datetime import date, timedelta

def ensure(p): Path(p).parent.mkdir(parents=True, exist_ok=True)

def mk_eia():
    d0 = date(2024, 1, 3)
    rows = []
    lvl = 420.0
    for i in range(10):
        dt = d0 + timedelta(days=7*i)
        lvl += np.random.randn()*1.0
        rows.append({"date": dt.isoformat(), "crude_stocks": round(lvl,2)})
    ensure("data/raw/eia/weekly_crude_stocks.csv")
    pd.DataFrame(rows).to_csv("data/raw/eia/weekly_crude_stocks.csv", index=False)

def mk_fred():
    d0 = date(2024, 1, 1)
    n = 30
    dates = [ (d0+timedelta(days=i)).isoformat() for i in range(n) ]
    for name, start in [("DTWEXBGS",103.0),("DGS10",4.1),("T10YIE",2.3)]:
        vals = np.linspace(start, start+0.2, n)
        ensure(f"data/raw/fred/{name}.csv")
        pd.DataFrame({"date":dates, "value":np.round(vals,4)}).to_csv(f"data/raw/fred/{name}.csv", index=False)

def mk_cftc():
    d0 = date(2024, 1, 3)
    rows = []
    for i in range(10):
        dt = d0 + timedelta(days=7*i)
        long = 300000 + 1000*i
        short = 250000 + 800*i
        oi = 1600000 + 500*i
        rows.append({"date": dt.isoformat(), "mm_net": long-short, "mm_gross_long": long, "mm_gross_short": short, "open_interest": oi})
    ensure("data/raw/cftc/managed_money.parquet")
    pd.DataFrame(rows).to_parquet("data/raw/cftc/managed_money.parquet", index=False)

def mk_cvol():
    d0 = date(2024, 1, 1)
    n = 30
    dates = [ (d0+timedelta(days=i)).isoformat() for i in range(n) ]
    df = pd.DataFrame({
        "date": dates,
        "cvol_1m": np.linspace(35, 40, n),
        "skew_25d": np.linspace(-4, -2, n),
        "term_premium": np.linspace(1.0, 1.3, n),
    })
    ensure("data/raw/cvol/wti_cvol.parquet")
    df.to_parquet("data/raw/cvol/wti_cvol.parquet", index=False)

def mk_chris():
    # Minimal CL1..CL3 continuous to satisfy _load_cme_continuous()
    for k in [1,2,3]:
        d0 = date(2024, 1, 1)
        n = 30
        dates = [ (d0+timedelta(days=i)).isoformat() for i in range(n) ]
        base = 70 + 0.5*k
        settle = np.round(base + np.sin(np.linspace(0,3,n)), 2)
        ensure(f"data/raw/chris/CL{k}.csv")
        pd.DataFrame({"date": dates, "settle": settle}).to_csv(f"data/raw/chris/CL{k}.csv", index=False)

if __name__ == "__main__":
    mk_eia(); mk_fred(); mk_cftc(); mk_cvol(); mk_chris()
    print("Seeded minimal raw data for CI.")
