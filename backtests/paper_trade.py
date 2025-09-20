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
