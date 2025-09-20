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
