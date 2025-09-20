#!/usr/bin/env bash
set -euo pipefail
ROOT="/Users/georgekurchey/CL_Model"
mkdir -p "$ROOT/models" "$ROOT/serve" "$ROOT/monitor" "$ROOT/scripts" "$ROOT/logs" "$ROOT/reports" "$ROOT/preds"

# -------- models/save_full_iso.py --------
cat > "$ROOT/models/save_full_iso.py" <<'PY'
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
import joblib

DEF_ROOT = Path("/Users/georgekurchey/CL_Model")
DEF_FEATURES = DEF_ROOT / "data/proc/features.csv"
DEF_MODELS = DEF_ROOT / "models" / "gbt_iso.joblib"

def parse_taus(s: str):
    parts = sorted(set(float(x) for x in s.split(",") if x.strip()))
    assert all(0.0 < t < 1.0 for t in parts)
    return parts

def load_features(path: Path):
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date").reset_index(drop=True)
    if "ret_1d" not in df.columns:
        raise SystemExit("ret_1d missing in features.")
    return df

def build_xy(df: pd.DataFrame):
    y = df["ret_1d"].astype(float).values
    X = df.drop(columns=["ret_1d","date"], errors="ignore")
    num = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
    return X[num].copy(), y, num

def fit_grid(Xtr, ytr, taus, params):
    models = {}
    for t in taus:
        m = GradientBoostingRegressor(loss="quantile",
                                      alpha=t,
                                      n_estimators=params["n_estimators"],
                                      learning_rate=params["learning_rate"],
                                      max_depth=params["max_depth"],
                                      min_samples_leaf=params["min_samples_leaf"],
                                      random_state=params["random_state"])
        m.fit(Xtr, ytr)
        models[t] = m
    return models

def iso_inverse(taus, y_cal, q_cal):
    taus = np.asarray(taus, float)
    cover = np.array([np.mean(y_cal <= q_cal[:,j]) for j,_ in enumerate(taus)])
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    fitted = iso.fit_transform(taus, cover)  # f(tau) ~ coverage
    xs = fitted.copy(); ys = taus.copy()
    for i in range(1, len(xs)):
        if xs[i] <= xs[i-1]: xs[i] = xs[i-1] + 1e-6
    def inv_f(t_star: float) -> float:
        t_star = float(np.clip(t_star, 0.0, 1.0))
        return float(np.interp(t_star, xs, ys))
    return inv_f, {"taus":taus.tolist(),"cover":cover.tolist(),"fitted":fitted.tolist()}

def main():
    ap = argparse.ArgumentParser(description="Train full GBT-ISO model (retrain if stale).")
    ap.add_argument("--features", default=str(DEF_FEATURES))
    ap.add_argument("--out", default=str(DEF_MODELS))
    ap.add_argument("--iso_days", type=int, default=252)
    ap.add_argument("--taus", type=str, default="0.05,0.15,0.25,0.35,0.50,0.65,0.75,0.85,0.95")
    ap.add_argument("--n_estimators", type=int, default=600)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--max_depth", type=int, default=3)
    ap.add_argument("--min_samples_leaf", type=int, default=60)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--refresh_days", type=int, default=30, help="retrain if model older than this")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    outp = Path(args.out)
    if outp.exists() and not args.force:
        age_days = (time.time() - outp.stat().st_mtime)/86400.0
        if age_days < args.refresh_days:
            print(f"[save_full_iso] Existing model is fresh ({age_days:.1f}d). SKIP retrain.")
            return

    df = load_features(Path(args.features))
    X, y, cols = build_xy(df)
    if len(df) <= args.iso_days + 250:
        raise SystemExit("Not enough rows for core+calibration. Reduce iso_days.")

    core_end = len(df) - args.iso_days
    Xcore, ycore = X.iloc[:core_end].values, y[:core_end]
    Xcal, ycal = X.iloc[core_end:].values, y[core_end:]

    imp = SimpleImputer(strategy="median")
    Xcore = imp.fit_transform(Xcore)
    Xcal  = imp.transform(Xcal)

    taus = parse_taus(args.taus)
    params = dict(n_estimators=args.n_estimators, learning_rate=args.learning_rate,
                  max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf,
                  random_state=args.random_state)
    models = fit_grid(Xcore, ycore, taus, params)

    q_cal = np.column_stack([models[t].predict(Xcal) for t in taus])
    q_cal = np.sort(q_cal, axis=1)
    inv_f, iso_diag = iso_inverse(taus, ycal, q_cal)

    t_eff = {"0.05": inv_f(0.05), "0.50": inv_f(0.50), "0.95": inv_f(0.95)}
    artifact = {
        "taus": taus,
        "cols": cols,
        "imputer": imp,
        "models": models,
        "t_eff": t_eff,
        "iso_diag": iso_diag,
        "meta": {"trained_at": pd.Timestamp.utcnow().isoformat(), "iso_days": args.iso_days, "params": params}
    }
    outp.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, outp)
    print(f"[save_full_iso] Saved -> {outp}")
if __name__ == "__main__":
    main()
PY

# -------- serve/predict_today.py --------
cat > "$ROOT/serve/predict_today.py" <<'PY'
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
PY

# -------- monitor/scoring_live.py --------
cat > "$ROOT/monitor/scoring_live.py" <<'PY'
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DEF_ROOT = Path("/Users/georgekurchey/CL_Model")
DEF_LIVE = DEF_ROOT / "preds/live_history.csv"
DEF_FEATS = DEF_ROOT / "data/proc/features.csv"
DEF_OUT = DEF_ROOT / "reports/monitor.html"

def crps_from_grid(y, q_grid, taus):
    taus = np.array(taus, float)
    w = np.empty_like(taus)
    w[0] = (taus[1]-0.0)/2
    w[-1] = (1.0-taus[-2])/2
    w[1:-1] = (taus[2:]-taus[:-2])/2
    L = []
    for i,t in enumerate(taus):
        e = y - q_grid[:,i]
        L.append(np.maximum(t*e, (t-1)*e))
    L = np.vstack(L).T
    return (L*w).sum(axis=1)

def main():
    ap = argparse.ArgumentParser(description="Update live monitoring (VaR exceedances & CRPS).")
    ap.add_argument("--history", default=str(DEF_LIVE))
    ap.add_argument("--features", default=str(DEF_FEATS))
    ap.add_argument("--out_html", default=str(DEF_OUT))
    args = ap.parse_args()

    if not Path(args.history).exists():
        raise SystemExit("No live_history.csv yet.")

    live = pd.read_csv(args.history, parse_dates=["date"]).sort_values("date")
    feats = pd.read_csv(args.features, parse_dates=["date"])[["date","ret_1d"]]
    df = live.merge(feats, on="date", how="left").rename(columns={"ret_1d":"y"})

    # derive tau grid from columns present
    tau_cols = sorted([c for c in df.columns if c.startswith("q") and len(c)==3 and c[1:].isdigit()],
                      key=lambda c: int(c[1:]))
    taus = [int(c[1:])/100.0 for c in tau_cols]
    # compute metrics only where y is available
    have_y = df["y"].notna().values
    y = df.loc[have_y, "y"].values
    q05 = df.loc[have_y, "q05"].values
    q95 = df.loc[have_y, "q95"].values
    q_grid = df.loc[have_y, tau_cols].values

    n = len(y)
    var5 = int((y < q05).sum())
    var95 = int((y > q95).sum())
    crps = crps_from_grid(y, q_grid, taus) if n>0 else np.array([])
    mean_crps = float(np.mean(crps)) if n>0 else float("nan")

    # rolling windows
    def roll_count(mask, win):
        x = pd.Series(mask.astype(int)).rolling(win).sum().iloc[-1]
        return int(x) if not np.isnan(x) else 0
    r30 = roll_count((df["y"] < df["q05"]).fillna(False).values, 30)
    r90 = roll_count((df["y"] < df["q05"]).fillna(False).values, 90)
    r180= roll_count((df["y"] < df["q05"]).fillna(False).values, 180)

    html = []
    html.append("<html><head><meta charset='utf-8'><title>CL Live Monitor</title>")
    html.append("<style>body{font-family:Arial,Helvetica,sans-serif;margin:20px;} table{border-collapse:collapse;} th,td{border:1px solid #ddd;padding:6px;} th{background:#f0f0f0;}</style></head><body>")
    html.append("<h2>CL_Model — Live Monitor</h2>")
    html.append(f"<p>Rows with realized y: {n}</p>")
    html.append("<ul>")
    html.append(f"<li>Mean CRPS (where realized): {mean_crps:.6f}</li>")
    html.append(f"<li>Total VaR 5% exceedances: {var5}</li>")
    html.append(f"<li>Total VaR 95% exceedances: {var95}</li>")
    html.append(f"<li>Rolling VaR5 exceedances: 30d={r30}, 90d={r90}, 180d={r180}</li>")
    html.append("</ul>")
    html.append("<h3>Latest predictions</h3>")
    html.append(df.tail(15).to_html(index=False))
    html.append("</body></html>")
    Path(args.out_html).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_html).write_text("\n".join(html), encoding="utf-8")
    print(f"[monitor] -> {args.out_html}")
if __name__ == "__main__":
    main()
PY

# -------- scripts/run_step4_daily.sh (progress+logs) --------
cat > "$ROOT/scripts/run_step4_daily.sh" <<'SH2'
#!/usr/bin/env bash
set -euo pipefail
ROOT="/Users/georgekurchey/CL_Model"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR" "$ROOT/reports" "$ROOT/preds"
TS=$(date +"%Y%m%d_%H%M%S")
LOG="$LOGDIR/step4_${TS}.log"
PROG="$LOGDIR/progress.json"
ln -sf "$LOG" "$LOGDIR/current.log"

progress () {
  # $1 stage, $2 pct
  printf '{"stage":"%s","pct":%s,"ts":"%s"}\n' "$1" "$2" "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" > "$PROG"
  echo "[$(date +"%F %T")] $1 ($2%)" | tee -a "$LOG"
}

err_trap () {
  progress "FAILED" 0
  echo "[ERROR] Step 4 failed. See log: $LOG"
}
trap err_trap ERR

# optional: load API keys
if [ -f "$ROOT/config/secrets.env" ]; then . "$ROOT/config/secrets.env"; fi

PY="$(command -v python3)"

progress "Start Step 4" 1

progress "Update CL strip (Yahoo)" 10
"$PY" "$ROOT/ingest/ndlink_cl.py" --source yahoo --m 12 --out "$ROOT/data/raw/cl_strip.parquet" --sleep 0.0 >> "$LOG" 2>&1

progress "Update Macro (FRED)" 20
"$PY" "$ROOT/ingest/fred_macro.py" --series DTWEXBGS DGS10 T10YIE DFII10 --out "$ROOT/data/raw/macro.parquet" >> "$LOG" 2>&1

progress "Update EIA weekly" 30
"$PY" "$ROOT/ingest/eia_weekly.py" --series WCESTUS1 WGTSTUS1 WDISTUS1 --out "$ROOT/data/raw/eia.parquet" >> "$LOG" 2>&1

progress "Rebuild features" 50
"$PY" "$ROOT/features/build_features_live.py" \
  --config "$ROOT/config/pipeline.json" \
  --raw_strip "$ROOT/data/raw/cl_strip.parquet" \
  --macro "$ROOT/data/raw/macro.parquet" \
  --vol "$ROOT/data/raw/vol.parquet" \
  --eia "$ROOT/data/raw/eia.parquet" \
  --out "$ROOT/data/proc/features.csv" >> "$LOG" 2>&1

progress "Train/refresh model if stale" 65
"$PY" "$ROOT/models/save_full_iso.py" --features "$ROOT/data/proc/features.csv" --refresh_days 30 >> "$LOG" 2>&1

progress "Predict today" 80
"$PY" "$ROOT/serve/predict_today.py" >> "$LOG" 2>&1

progress "Update live monitor" 90
"$PY" "$ROOT/monitor/scoring_live.py" >> "$LOG" 2>&1

progress "DONE" 100
echo "Done. Logs: $LOG ; Monitor: $ROOT/reports/monitor.html"
SH2

# -------- scripts/progress_watch.sh --------
cat > "$ROOT/scripts/progress_watch.sh" <<'SH3'
#!/usr/bin/env bash
ROOT="/Users/georgekurchey/CL_Model"
LOGDIR="$ROOT/logs"
PROG="$LOGDIR/progress.json"
LOG="$LOGDIR/current.log"
echo "Watching progress at $PROG"
while true; do
  if [ -f "$PROG" ]; then
    cat "$PROG"
  else
    echo '{"stage":"(waiting)","pct":0}'
  fi
  sleep 2
done
SH3

chmod +x "$ROOT/scripts/run_step4_daily.sh" "$ROOT/scripts/progress_watch.sh"
echo "Installed Step 4: runner + watcher + model/predict/monitor."
