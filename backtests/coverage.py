
import pandas as pd, numpy as np
from pathlib import Path

def load_targets():
    feat = pd.read_parquet("data/proc/features.parquet")
    if "date" not in feat.columns or "ret_1d" not in feat.columns:
        raise SystemExit("features.parquet missing required columns: date, ret_1d")
    y = feat[["date","ret_1d"]].dropna().copy()
    y["date"] = pd.to_datetime(y["date"])
    return y

def read_preds():
    pred_files = sorted(Path("preds").glob("*.parquet"))
    out = {}
    for p in pred_files:
        df = pd.read_parquet(p)
        if "date" not in df.columns:
            continue
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        out[p.stem] = df
    if not out:
        raise SystemExit("no prediction files found under preds/")
    return out

def pick_q(df, tau):
    import re
    want = int(round(tau*100))
    name_tau = f"tau_{want:02d}"
    name_q   = f"q{want:02d}"
    if name_tau in df.columns:
        return df[name_tau]
    if name_q in df.columns:
        return df[name_q]
    qcols = [c for c in df.columns if c.startswith("tau_") or re.fullmatch(r"q\d{2}", c)]
    if not qcols:
        raise KeyError(f"no quantile columns like 'qXX' or 'tau_XX' in {list(df.columns)[:10]}")
    def to_pct(c):
        return int(c.split("_")[1]) if c.startswith("tau_") else int(c[1:])
    nearest = min(qcols, key=lambda c: abs(to_pct(c) - want))
    return df[nearest]

    qcols = [c for c in df.columns if c.startswith("tau_")]
    if not qcols:
        raise KeyError(f"no quantile columns like 'tau_XX' in {list(df.columns)[:10]}")
    taus = [int(c.split("_")[1]) / 100.0 for c in qcols]
    idx = np.argmin(np.abs(np.array(taus) - tau))
    return df[qcols[idx]]

def coverage_for(model_df, y):
    df = y.merge(model_df, on="date", how="inner")
    if len(df) == 0:
        return np.nan, np.nan
    q05 = pick_q(df, 0.05)
    q95 = pick_q(df, 0.95)
    # Coverage definitions: fraction below q95 should be ~0.95; fraction below q05 ~0.05
    cov95 = float((df["ret_1d"] <= q95).mean())
    cov05 = float((df["ret_1d"] <= q05).mean())
    return cov05, cov95

def main():
    y = load_targets()
    preds = read_preds()
    rows = []
    for name, df in preds.items():
        cov05, cov95 = coverage_for(df, y)
        rows.append({"model": name, "VaR05_cov": cov05, "VaR95_cov": cov95})
    cov = pd.DataFrame(rows).sort_values("model")
    Path("reports").mkdir(parents=True, exist_ok=True)
    cov.to_csv("reports/coverage.csv", index=False)
    print("reports/coverage.csv")
    # CI guard: require reasonably close upper-tail coverage
    # (we use a soft floor; tune as needed)
    cov95 = float(cov["VaR95_cov"].mean())
    if cov95 < 0.90:
        print(f"FAIL: VaR95 coverage too low: {cov95:.3f}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
