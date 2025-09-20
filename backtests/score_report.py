import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DEF_PREDS = Path("/Users/georgekurchey/CL_Model/preds")
DEF_OUT = Path("/Users/georgekurchey/CL_Model/reports/report.html")

def pinball_loss(y, q, tau):
    e = y - q
    return np.maximum(tau*e, (tau-1)*e)

def crps_from_grid(y, q_grid, taus):
    taus = np.array(taus, dtype=float)
    w = np.empty_like(taus)
    w[0] = (taus[1] - 0.0)/2
    w[-1] = (1.0 - taus[-2])/2
    w[1:-1] = (taus[2:] - taus[:-2]) / 2
    losses = [pinball_loss(y, q_grid[:,i], t) for i,t in enumerate(taus)]
    L = np.vstack(losses).T
    return (L * w).sum(axis=1)

def main():
    ap = argparse.ArgumentParser(description="Score preds and create HTML report.")
    ap.add_argument("--preds_dir", default=str(DEF_PREDS))
    ap.add_argument("--out_html", default=str(DEF_OUT))
    args = ap.parse_args()

    p = Path(args.preds_dir)
    files = sorted(p.glob("preds_fold*.csv"))
    if not files:
        raise SystemExit(f"No preds_fold*.csv in {p}")

    dfs = [pd.read_csv(f, parse_dates=["date"]) for f in files]
    df = pd.concat(dfs, axis=0, ignore_index=True).sort_values("date").reset_index(drop=True)

    tau_cols = sorted([c for c in df.columns if c.startswith("q") and len(c)==3 and c[1:].isdigit()],
                      key=lambda c: int(c[1:]))
    taus = [int(c[1:])/100.0 for c in tau_cols]

    y = df["y"].values
    q05, q95 = df["q05"].values, df["q95"].values
    var5_ex = int((y < q05).sum())
    var95_ex = int((y > q95).sum())
    n = len(df)

    q_grid = df[tau_cols].values
    crps = crps_from_grid(y, q_grid, taus)
    mean_crps = float(np.mean(crps))

    pf = (df.groupby("fold")
            .agg(n=("y","size"),
                 var5=("y", lambda s: int((s.values < df.loc[s.index, "q05"].values).sum())),
                 var95=("y", lambda s: int((s.values > df.loc[s.index, "q95"].values).sum())),
                 crps=("y", lambda s: float(np.mean(crps[s.index]))))
            .reset_index())

    html = []
    html.append("<html><head><meta charset='utf-8'><title>CL_Model Report</title>")
    html.append("<style>body{font-family:Arial,Helvetica,sans-serif;margin:20px;} table{border-collapse:collapse;} th,td{border:1px solid #ddd;padding:6px;} th{background:#f0f0f0;}</style>")
    html.append("</head><body>")
    html.append("<h2>CL_Model — Walk-forward Report</h2>")
    html.append(f"<p>Total rows: {n}</p>")
    html.append("<h3>Overall</h3><ul>")
    html.append(f"<li>Mean CRPS: {mean_crps:.6f}</li>")
    html.append(f"<li>VaR 5% exceedances: {var5_ex}/{n} (target ≈ 5%)</li>")
    html.append(f"<li>VaR 95% exceedances: {var95_ex}/{n} (target ≈ 5%)</li>")
    html.append("</ul>")
    html.append("<h3>Per-fold summary</h3>")
    html.append(pf.to_html(index=False))
    html.append("<h3>Sample predictions</h3>")
    html.append(df[["date","fold","y","q05","q50","q95"]].tail(15).to_html(index=False))
    html.append("</body></html>")

    out = Path(args.out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(html), encoding="utf-8")
    print(f"Report -> {out}")
    print(f"Mean CRPS: {mean_crps:.6f}")
    print(f"VaR 5% exceedances: {var5_ex}/{n}")
    print(f"VaR 95% exceedances: {var95_ex}/{n}")

if __name__ == "__main__":
    main()
