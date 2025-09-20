#!/usr/bin/env bash
set -euo pipefail
ROOT="/Users/georgekurchey/CL_Model"
mkdir -p "$ROOT/signals" "$ROOT/backtests" "$ROOT/config" "$ROOT/scripts" "$ROOT/reports"

# ---------- config/signal.json ----------
cat > "$ROOT/config/signal.json" <<'JSON'
{
  "lookback_vol_days": 60,
  "target_vol_daily": 0.01,
  "scale_k": 0.75,
  "leverage_cap": 2.0,
  "cost_per_turnover": 0.0002,
  "use_grid_for_scale": true
}
JSON

# ---------- signals/generate_signal.py ----------
cat > "$ROOT/signals/generate_signal.py" <<'PY'
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/georgekurchey/CL_Model")
DEF_CFG = ROOT / "config/signal.json"
DEF_FEATS = ROOT / "data/proc/features.csv"
DEF_LIVE = ROOT / "preds/live_today.csv"
DEF_HIST = ROOT / "signals/signal_history.csv"
DEF_OUT  = ROOT / "signals/live_signal.csv"

def interp_tau_at_zero(qvals, taus):
    x = np.array(qvals, float)
    y = np.array(taus, float)
    ord = np.argsort(x)
    x = x[ord]; y = y[ord]
    x[0] -= 1e-8; x[-1] += 1e-8
    return float(np.interp(0.0, x, y))

def main():
    ap = argparse.ArgumentParser(description="Generate daily signal & position from quantiles.")
    ap.add_argument("--config", default=str(DEF_CFG))
    ap.add_argument("--features", default=str(DEF_FEATS))
    ap.add_argument("--live_pred", default=str(DEF_LIVE))
    ap.add_argument("--history", default=str(DEF_HIST))
    ap.add_argument("--out", default=str(DEF_OUT))
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    feats = pd.read_csv(args.features, parse_dates=["date"]).sort_values("date")
    if not Path(args.live_pred).exists():
        raise SystemExit("live_today.csv not found. Run Step 3b/4 first.")

    live = pd.read_csv(args.live_pred, parse_dates=["date"]).iloc[-1]
    date = pd.to_datetime(live["date"]).tz_localize(None)

    # collect quantiles grid
    tau_cols = sorted([c for c in live.index if c.startswith("q") and len(c)==3 and c[1:].isdigit()],
                      key=lambda c: int(c[1:]))
    taus = [int(c[1:])/100.0 for c in tau_cols]
    q_grid = np.array([float(live[c]) for c in tau_cols], float)
    q05 = float(live.get("q05", q_grid[0] if len(q_grid) else np.nan))
    q50 = float(live.get("q50", np.median(q_grid) if len(q_grid) else np.nan))
    q95 = float(live.get("q95", q_grid[-1] if len(q_grid) else np.nan))

    # realized volatility estimate (exclude last NaN ret if any)
    y = feats["ret_1d"].astype(float)
    y_look = y.dropna().iloc[-cfg["lookback_vol_days"]:] if y.notna().sum()>=5 else y.dropna()
    sigma = float(y_look.std(ddof=0)) if len(y_look) > 0 else 0.01

    # scale for edge: use half-spread of predictive quantiles if available, else sigma
    if cfg.get("use_grid_for_scale", True) and len(q_grid)>=3:
        half_spread = max(1e-6, 0.5 * (q95 - q05))
        scale = half_spread
    else:
        scale = max(1e-6, sigma)

    # standardized edge from median
    s = float(np.clip(q50 / scale, -5.0, 5.0))
    # raw position from edge
    pos_edge = cfg["scale_k"] * s
    # risk scaling to target daily vol
    risk_scale = cfg["target_vol_daily"] / max(1e-6, sigma)
    pos = float(np.clip(pos_edge * risk_scale, -cfg["leverage_cap"], cfg["leverage_cap"]))

    # probability of up-move (approx via CDF at 0 using grid)
    p_up = np.nan
    if len(q_grid) >= 3:
        tau0 = interp_tau_at_zero(q_grid, taus)
        p_up = 1.0 - tau0

    rec = {
        "date": date,
        "q05": q05, "q50": q50, "q95": q95,
        "sigma_lookback": sigma,
        "edge_std": s,
        "pos_raw": pos_edge,
        "pos": pos,
        "p_up": p_up
    }

    # write live + append history
    out_df = pd.DataFrame([rec])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    hist_p = Path(args.history)
    if hist_p.exists():
        hist = pd.read_csv(hist_p, parse_dates=["date"])
        hist = pd.concat([hist, out_df], axis=0, ignore_index=True)
        hist = hist.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    else:
        hist = out_df.copy()
    hist.to_csv(hist_p, index=False)
    print(f"[signals] live -> {args.out} ; appended -> {args.history} ; pos={pos:.3f}")
if __name__ == "__main__":
    main()
PY

# ---------- backtests/paper_trade.py ----------
cat > "$ROOT/backtests/paper_trade.py" <<'PY'
import argparse, json, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path("/Users/georgekurchey/CL_Model")
DEF_CFG = ROOT / "config/signal.json"
DEF_FEATS = ROOT / "data/proc/features.csv"
DEF_SIGS = ROOT / "signals/signal_history.csv"
DEF_OUT_HTML = ROOT / "reports/signal_dashboard.html"

def max_drawdown(equity):
    peaks = np.maximum.accumulate(equity)
    dd = (equity - peaks) / peaks
    return float(dd.min()) if len(dd) else 0.0

def sharpe(returns, ann=252):
    m = np.mean(returns); s = np.std(returns, ddof=0)
    if s == 0: return 0.0
    return float((m / s) * np.sqrt(ann))

def main():
    ap = argparse.ArgumentParser(description="Paper-trade backtest for daily signal.")
    ap.add_argument("--config", default=str(DEF_CFG))
    ap.add_argument("--features", default=str(DEF_FEATS))
    ap.add_argument("--signals", default=str(DEF_SIGS))
    ap.add_argument("--out_html", default=str(DEF_OUT_HTML))
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    feats = pd.read_csv(args.features, parse_dates=["date"]).sort_values("date")
    if not Path(args.signals).exists():
        raise SystemExit("signals/signal_history.csv not found. Run generate_signal first.")
    sig = pd.read_csv(args.signals, parse_dates=["date"]).sort_values("date")

    df = feats.merge(sig[["date","pos"]], on="date", how="left")
    df["pos"] = df["pos"].ffill().fillna(0.0)

    # Strategy uses yesterday's position on today's return
    df["pos_prev"] = df["pos"].shift(1).fillna(0.0)
    r = df["ret_1d"].astype(float).fillna(0.0).values
    pos_prev = df["pos_prev"].values

    # Turnover & costs
    delta = np.diff(df["pos"].fillna(0.0).values, prepend=0.0)
    turnover = np.abs(delta)
    cost = cfg.get("cost_per_turnover", 0.0) * turnover

    strat_ret = pos_prev * r - cost
    equity = (1.0 + pd.Series(strat_ret)).cumprod().values

    stats = {
        "days": int(len(strat_ret)),
        "mean_daily": float(np.mean(strat_ret)),
        "vol_daily": float(np.std(strat_ret, ddof=0)),
        "sharpe": sharpe(strat_ret),
        "max_drawdown": max_drawdown(equity),
        "hit_ratio": float(np.mean(strat_ret > 0.0)) if len(strat_ret) else 0.0,
        "avg_turnover": float(np.mean(turnover))
    }

    # Simple HTML
    html = []
    html.append("<html><head><meta charset='utf-8'><title>CL Signals Dashboard</title>")
    html.append("<style>body{font-family:Arial,Helvetica,sans-serif;margin:20px;} table{border-collapse:collapse;} th,td{border:1px solid #ddd;padding:6px;} th{background:#f0f0f0;}</style>")
    html.append("</head><body>")
    html.append("<h2>CL_Model — Signals & Paper-Trade</h2>")
    html.append("<h3>Key stats</h3><ul>")
    for k,v in stats.items():
        if k in ("sharpe","max_drawdown","mean_daily","vol_daily","avg_turnover"):
            html.append(f"<li>{k}: {v:.4f}</li>")
        else:
            html.append(f"<li>{k}: {v}</li>")
    html.append("</ul>")
    tail = pd.DataFrame({
        "date": df["date"].tail(15).dt.strftime("%Y-%m-%d"),
        "pos_prev": df["pos_prev"].tail(15),
        "ret_1d": df["ret_1d"].tail(15),
        "strat_ret": pd.Series(strat_ret).tail(15)
    })
    html.append("<h3>Latest 15 days</h3>")
    html.append(tail.to_html(index=False))
    html.append("</body></html>")

    outp = Path(args.out_html)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(html), encoding="utf-8")
    print(f"[paper] -> {outp}")
    print("[paper] stats:", stats)

if __name__ == "__main__":
    main()
PY

# ---------- scripts/run_step6_signal.sh ----------
cat > "$ROOT/scripts/run_step6_signal.sh" <<'SH2'
#!/usr/bin/env bash
set -euo pipefail
ROOT="/Users/georgekurchey/CL_Model"
[ -f "$ROOT/.venv/bin/activate" ] && . "$ROOT/.venv/bin/activate" || true
PY="$(command -v python3)"

# Generate today's signal & append history
"$PY" "$ROOT/signals/generate_signal.py" --config "$ROOT/config/signal.json"

# Update paper-trade dashboard
"$PY" "$ROOT/backtests/paper_trade.py" --config "$ROOT/config/signal.json" --out_html "$ROOT/reports/signal_dashboard.html"

echo "Signal ready -> $ROOT/signals/live_signal.csv"
echo "History -> $ROOT/signals/signal_history.csv"
echo "Dashboard -> $ROOT/reports/signal_dashboard.html"
SH2

chmod +x "$ROOT/scripts/run_step6_signal.sh"
echo "Installed Step 6 (signals & paper-trade)."
