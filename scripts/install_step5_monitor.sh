#!/usr/bin/env bash
set -euo pipefail
ROOT="/Users/georgekurchey/CL_Model"
mkdir -p "$ROOT/monitor" "$ROOT/config" "$ROOT/scripts" "$ROOT/logs" "$ROOT/reports"

# ---------- config/monitor.json ----------
cat > "$ROOT/config/monitor.json" <<'JSON'
{
  "windows": { "short": 30, "medium": 90, "long": 180 },
  "var_target": 0.05,
  "var_z": 1.96,
  "crps_pct_worsen": 0.20,
  "psi_warn": 0.10,
  "psi_alert": 0.25,
  "freshness_days": { "features": 2, "cl_strip": 2, "macro": 3, "eia": 10 }
}
JSON

# ---------- monitor/metrics.py ----------
cat > "$ROOT/monitor/metrics.py" <<'PY'
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/georgekurchey/CL_Model")
DEF_CONFIG = ROOT / "config/monitor.json"
DEF_FEATURES = ROOT / "data/proc/features.csv"
DEF_LIVE = ROOT / "preds/live_history.csv"
DEF_OUT = ROOT / "reports/monitor_metrics.json"

def wilson_interval(k, n, z=1.96):
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    p_hat = k / n
    denom = 1 + z*z/n
    center = (p_hat + z*z/(2*n)) / denom
    half = (z/denom) * math.sqrt( (p_hat*(1-p_hat)/n) + (z*z/(4*n*n)) )
    return (p_hat, max(0.0, center - half), min(1.0, center + half))

def crps_from_grid(y, q_grid, taus):
    taus = np.array(taus, dtype=float)
    w = np.empty_like(taus)
    w[0] = (taus[1] - 0.0)/2
    w[-1] = (1.0 - taus[-2])/2
    w[1:-1] = (taus[2:] - taus[:-2]) / 2
    L = []
    for i,t in enumerate(taus):
        e = y - q_grid[:,i]
        L.append(np.maximum(t*e, (t-1)*e))
    L = np.vstack(L).T
    return (L*w).sum(axis=1)

def psi(actual: np.ndarray, expected: np.ndarray, bins: int = 10):
    """Population Stability Index for numeric 1D arrays."""
    a = pd.Series(actual).replace([np.inf,-np.inf], np.nan).dropna()
    e = pd.Series(expected).replace([np.inf,-np.inf], np.nan).dropna()
    if len(a) < 5 or len(e) < 50:
        return float("nan")
    qs = np.linspace(0, 1, bins+1)
    edges = np.quantile(e, qs)
    edges = np.unique(edges)
    if len(edges) < 3:
        return float("nan")
    a_hist = np.histogram(a, bins=edges)[0].astype(float)
    e_hist = np.histogram(e, bins=edges)[0].astype(float)
    a_pct = np.maximum(a_hist / max(1,a_hist.sum()), 1e-6)
    e_pct = np.maximum(e_hist / max(1,e_hist.sum()), 1e-6)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))

def main():
    ap = argparse.ArgumentParser(description="Compute monitoring metrics (VaR coverage, CRPS, PSI, freshness).")
    ap.add_argument("--config", default=str(DEF_CONFIG))
    ap.add_argument("--features", default=str(DEF_FEATURES))
    ap.add_argument("--live", default=str(DEF_LIVE))
    ap.add_argument("--out", default=str(DEF_OUT))
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())

    # Load data
    feats = pd.read_csv(args.features, parse_dates=["date"]).sort_values("date")
    if not Path(args.live).exists():
        raise SystemExit("live_history.csv not found. Run Step 3b / Step 4 first.")
    live = pd.read_csv(args.live, parse_dates=["date"]).sort_values("date")

    # Merge realized returns
    df = live.merge(feats[["date","ret_1d"]], on="date", how="left").rename(columns={"ret_1d":"y"})
    # infer tau grid
    tau_cols = sorted([c for c in df.columns if c.startswith("q") and len(c)==3 and c[1:].isdigit()],
                      key=lambda c: int(c[1:]))
    taus = [int(c[1:])/100.0 for c in tau_cols]

    # coverage calculations where y is known
    avail = df["y"].notna().values
    df_av = df.loc[avail].copy()
    y = df_av["y"].values
    q05 = df_av["q05"].values if "q05" in df_av else df_av[tau_cols[0]].values
    q95 = df_av["q95"].values if "q95" in df_av else df_av[tau_cols[-1]].values
    q_grid = df_av[tau_cols].values

    n = len(df_av)
    k5 = int((y < q05).sum())
    k95 = int((y > q95).sum())
    p5, lo5, hi5 = wilson_interval(k5, n, z=cfg.get("var_z",1.96))
    p95, lo95, hi95 = wilson_interval(k95, n, z=cfg.get("var_z",1.96))

    # rolling windows
    wins = cfg["windows"]
    roll = {}
    for name, win in wins.items():
        if n >= win:
            k5w = int((y[-win:] < q05[-win:]).sum())
            k95w = int((y[-win:] > q95[-win:]).sum())
            p5w, lo5w, hi5w = wilson_interval(k5w, win, z=cfg.get("var_z",1.96))
            p95w, lo95w, hi95w = wilson_interval(k95w, win, z=cfg.get("var_z",1.96))
            roll[name] = {
                "n": win,
                "var5": {"k": k5w, "p": p5w, "lo": lo5w, "hi": hi5w},
                "var95": {"k": k95w, "p": p95w, "lo": lo95w, "hi": hi95w}
            }
        else:
            roll[name] = None

    # CRPS trend
    crps = crps_from_grid(y, q_grid, taus) if n>0 else np.array([])
    crps_short = float(np.mean(crps[-wins["short"]:])) if n >= wins["short"] else float("nan")
    crps_base = float(np.mean(crps[max(0, n-wins["long"]-wins["short"]):max(0, n-wins["short"])])) if n > wins["short"] else float("nan")
    crps_drift = (crps_short - crps_base) / crps_base if (not math.isnan(crps_short) and not math.isnan(crps_base) and crps_base!=0.0) else float("nan")

    # PSI drift (last 30d vs prior 252d) on numeric features
    numeric_cols = [c for c in feats.columns if c not in ("date","ret_1d") and np.issubdtype(feats[c].dtype, np.number)]
    psi_map = {}
    if len(feats) > 400:
        ref = feats.iloc[-(wins["long"]+252):-wins["long"]][numeric_cols] if len(feats) > (wins["long"]+252) else feats.iloc[:-wins["long"]][numeric_cols]
        cur = feats.iloc[-wins["short"]:,][numeric_cols]
        for c in numeric_cols:
            psi_map[c] = psi(cur[c].values, ref[c].values)
    else:
        psi_map = {c: float("nan") for c in numeric_cols}

    # Freshness
    def latest_date_csv(p): 
        return pd.read_csv(p, parse_dates=["date"])["date"].max() if Path(p).exists() else pd.NaT
    latest = {
        "features": feats["date"].max(),
        "live_pred": live["date"].max(),
        "cl_strip": latest_date_csv(ROOT/"data/raw/cl_strip.parquet"),
        "macro": latest_date_csv(ROOT/"data/raw/macro.parquet"),
        "eia": latest_date_csv(ROOT/"data/raw/eia.parquet")
    }
    latest_str = {k: (str(v.date()) if not pd.isna(v) else None) for k,v in latest.items()}

    out = {
        "asof": str(pd.Timestamp.utcnow()),
        "counts": {"n_realized": n},
        "coverage_total": {"n": n, "var5": {"k": k5, "p": p5, "lo": lo5, "hi": hi5}, "var95": {"k": k95, "p": p95, "lo": lo95, "hi": hi95}},
        "coverage_roll": roll,
        "crps": {"short_mean": crps_short, "baseline_mean": crps_base, "pct_change": crps_drift},
        "psi": psi_map,
        "latest_dates": latest_str
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[metrics] -> {args.out}")
PY

# ---------- monitor/alerts.py ----------
cat > "$ROOT/monitor/alerts.py" <<'PY'
import argparse, json, os, smtplib, ssl
from email.mime.text import MIMEText
from pathlib import Path
import requests

ROOT = Path("/Users/georgekurchey/CL_Model")
DEF_CONFIG = ROOT / "config/monitor.json"
DEF_METRICS = ROOT / "reports/monitor_metrics.json"
DEF_ALERTS_JSON = ROOT / "reports/alerts_last.json"

def within_band(p, lo, hi):
    return (p >= lo) and (p <= hi)

def load_env():
    return {
      "SLACK_WEBHOOK_URL": os.environ.get("SLACK_WEBHOOK_URL","").strip(),
      "SMTP_HOST": os.environ.get("SMTP_HOST","").strip(),
      "SMTP_PORT": int(os.environ.get("SMTP_PORT","587")),
      "SMTP_USER": os.environ.get("SMTP_USER","").strip(),
      "SMTP_PASS": os.environ.get("SMTP_PASS","").strip(),
      "ALERT_EMAIL_FROM": os.environ.get("ALERT_EMAIL_FROM","").strip(),
      "ALERT_EMAIL_TO": os.environ.get("ALERT_EMAIL_TO","").strip()
    }

def send_slack(msg, hook):
    try:
        r = requests.post(hook, json={"text": msg}, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"[alerts] Slack send failed: {e}")
        return False

def send_email(subject, msg, env):
    try:
        if not env["SMTP_HOST"] or not env["ALERT_EMAIL_TO"]:
            return False
        ctx = ssl.create_default_context()
        with smtplib.SMTP(env["SMTP_HOST"], env["SMTP_PORT"]) as s:
            s.starttls(context=ctx)
            if env["SMTP_USER"]:
                s.login(env["SMTP_USER"], env["SMTP_PASS"])
            m = MIMEText(msg, "plain", "utf-8")
            m["Subject"] = subject
            m["From"] = env["ALERT_EMAIL_FROM"] or env["SMTP_USER"]
            m["To"] = env["ALERT_EMAIL_TO"]
            s.sendmail(m["From"], env["ALERT_EMAIL_TO"].split(","), m.as_string())
        return True
    except Exception as e:
        print(f"[alerts] Email send failed: {e}")
        return False

def main():
    ap = argparse.ArgumentParser(description="Evaluate metrics and send alerts.")
    ap.add_argument("--config", default=str(DEF_CONFIG))
    ap.add_argument("--metrics", default=str(DEF_METRICS))
    ap.add_argument("--out_json", default=str(DEF_ALERTS_JSON))
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    met = json.loads(Path(args.metrics).read_text())
    env = load_env()

    msgs = []
    fired = []

    # 1) VaR coverage (use medium window if available, else total)
    var_target = cfg.get("var_target", 0.05)
    medium = met["coverage_roll"].get("medium")
    scope = medium if medium else met["coverage_total"]
    n = scope["n"] if "n" in scope else scope["var5"]["k"] + scope["var95"]["k"]
    p5, lo5, hi5 = scope["var5"]["p"], scope["var5"]["lo"], scope["var5"]["hi"]
    p95, lo95, hi95 = scope["var95"]["p"], scope["var95"]["lo"], scope["var95"]["hi"]

    if not within_band(p5, lo5, hi5):
        fired.append("VAR5_COVERAGE_OUT_OF_BAND")
        msgs.append(f"VaR 5% coverage out of band (n={scope['n'] if 'n' in scope else 'NA'}): p={p5:.3f}, band=({lo5:.3f},{hi5:.3f})")

    if not within_band(p95, lo95, hi95):
        fired.append("VAR95_COVERAGE_OUT_OF_BAND")
        msgs.append(f"VaR 95% coverage out of band (n={scope['n'] if 'n' in scope else 'NA'}): p={p95:.3f}, band=({lo95:.3f},{hi95:.3f})")

    # 2) CRPS drift
    drift = met["crps"]["pct_change"]
    if isinstance(drift, (float,int)) and not (drift is None) and drift == drift:  # not NaN
        if drift > cfg.get("crps_pct_worsen", 0.20):
            fired.append("CRPS_WORSENING")
            msgs.append(f"CRPS degraded: short mean is {drift*100:.1f}% worse than baseline.")

    # 3) PSI drift
    psi_warn = cfg.get("psi_warn", 0.10)
    psi_alert = cfg.get("psi_alert", 0.25)
    psi_breaches = {k:v for k,v in met["psi"].items() if isinstance(v, (float,int)) and v==v and v>=psi_alert}
    psi_warns = {k:v for k,v in met["psi"].items() if isinstance(v, (float,int)) and v==v and psi_warn <= v < psi_alert}
    if psi_breaches:
        fired.append("PSI_ALERT")
        top = sorted(psi_breaches.items(), key=lambda kv: -kv[1])[:10]
        msgs.append("High feature drift (PSI>=0.25): " + ", ".join([f"{k}:{v:.2f}" for k,v in top]))
    if psi_warns:
        fired.append("PSI_WARN")
        top = sorted(psi_warns.items(), key=lambda kv: -kv[1])[:10]
        msgs.append("Moderate feature drift (PSI 0.10–0.25): " + ", ".join([f"{k}:{v:.2f}" for k,v in top]))

    # 4) Data freshness
    fresh_cfg = cfg.get("freshness_days", {})
    stale_msgs = []
    for key, days in fresh_cfg.items():
        dt_str = met["latest_dates"].get(key)
        if not dt_str:
            stale_msgs.append(f"{key}: missing")
            continue
        # no exact "days since" calc (no tz info); flag handled in Step 4 cadence
    if stale_msgs:
        fired.append("FRESHNESS_WARN")
        msgs.append("Data freshness issues: " + "; ".join(stale_msgs))

    # Compose and send
    summary = " | ".join(msgs) if msgs else "All metrics within control bands."
    out = {"asof": met["asof"], "fired": fired, "messages": msgs, "source_metrics": str(Path(args.metrics))}
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"[alerts] -> {args.out_json}")
    print("[alerts] " + summary)

    if msgs:
        text = "CL_Model Alerts\n\n" + "\n".join(f"- {m}" for m in msgs)
        sent = False
        if env["SLACK_WEBHOOK_URL"]:
            sent = send_slack(text, env["SLACK_WEBHOOK_URL"]) or sent
        sent = send_email("CL_Model Alerts", text, env) or sent
        if not sent:
            print("[alerts] No Slack webhook or email SMTP configured; alerts logged only.")

if __name__ == "__main__":
    main()
PY

# ---------- scripts/run_step5_monitor.sh ----------
cat > "$ROOT/scripts/run_step5_monitor.sh" <<'SH2'
#!/usr/bin/env bash
set -euo pipefail
ROOT="/Users/georgekurchey/CL_Model"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"
TS=$(date +"%Y%m%d_%H%M%S")
LOG="$LOGDIR/step5_${TS}.log"
echo "[Step5] Start $(date)" | tee -a "$LOG"

# Load secrets for Slack/Email if present
[ -f "$ROOT/config/secrets.env" ] && . "$ROOT/config/secrets.env" || true
[ -f "$ROOT/.venv/bin/activate" ] && . "$ROOT/.venv/bin/activate" || true
PY="$(command -v python3)"

# 1) Compute metrics JSON
echo "[Step5] metrics.py ..." | tee -a "$LOG"
"$PY" "$ROOT/monitor/metrics.py" --config "$ROOT/config/monitor.json" --out "$ROOT/reports/monitor_metrics.json" 2>&1 | tee -a "$LOG"

# 2) Evaluate + send alerts
echo "[Step5] alerts.py ..." | tee -a "$LOG"
"$PY" "$ROOT/monitor/alerts.py" --config "$ROOT/config/monitor.json" --metrics "$ROOT/reports/monitor_metrics.json" 2>&1 | tee -a "$LOG"

echo "[Step5] Done. See $LOG"
SH2

chmod +x "$ROOT/scripts/run_step5_monitor.sh"
echo "Installed Step 5 monitoring & alerts."
