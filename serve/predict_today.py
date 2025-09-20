import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

DEF_ROOT = Path("/Users/georgekurchey/CL_Model")
DEF_FEATURES = DEF_ROOT / "data/proc/features.csv"
DEF_MODEL = DEF_ROOT / "models/gbt_iso.joblib"
DEF_OUT = DEF_ROOT / "preds/live_today.csv"
DEF_HISTORY = DEF_ROOT / "preds/live_history.csv"

def interp(row_q, taus, tstar):
    return float(np.interp(tstar, taus, row_q))

def main():
    ap = argparse.ArgumentParser(description="Predict today's quantiles using saved GBT-ISO model.")
    ap.add_argument("--features", default=str(DEF_FEATURES))
    ap.add_argument("--model", default=str(DEF_MODEL))
    ap.add_argument("--out", default=str(DEF_OUT))
    ap.add_argument("--history", default=str(DEF_HISTORY))
    args = ap.parse_args()

    art = joblib.load(Path(args.model))
    taus = np.array(art["taus"], float)
    cols = art["cols"]
    imp = art["imputer"]
    models = art["models"]
    t_eff = art["t_eff"]

    df = pd.read_csv(args.features, parse_dates=["date"]).sort_values("date")
    row = df.iloc[-1].copy()
    X = df[cols].iloc[[-1]].values
    X = imp.transform(X)

    q_grid = np.column_stack([models[t].predict(X) for t in taus])[0]
    q_grid = np.sort(q_grid)  # enforce monotonic
    q05 = interp(q_grid, taus, float(t_eff["0.05"]))
    q50 = interp(q_grid, taus, float(t_eff["0.50"]))
    q95 = interp(q_grid, taus, float(t_eff["0.95"]))

    out_cols = {f"q{int(t*100):02d}": v for t,v in zip(taus, q_grid)}
    rec = {"date": row["date"], "q05": q05, "q50": q50, "q95": q95, **out_cols}
    out = pd.DataFrame([rec])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    # append to history
    hist = Path(args.history)
    if hist.exists():
        dh = pd.read_csv(hist, parse_dates=["date"])
        dh = pd.concat([dh, out], axis=0, ignore_index=True)
        dh = dh.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    else:
        dh = out.copy()
    dh.to_csv(hist, index=False)
    print(f"[predict_today] Wrote -> {args.out} and appended -> {args.history}")
if __name__ == "__main__":
    main()
