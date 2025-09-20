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
