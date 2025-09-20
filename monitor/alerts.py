import argparse, json, sys
from pathlib import Path

ROOT = Path("/Users/georgekurchey/CL_Model")
MMET = ROOT/"reports/monitor_metrics.json"
CFG  = ROOT/"config/monitor.json"

def severity_rank(s):
    return {"CRITICAL":3, "WARN":2, "OK":1, "UNKNOWN":0}.get(s, 0)

def gather_metrics():
    if not MMET.exists():
        raise SystemExit("monitor_metrics.json not found; run metrics first.")
    m = json.loads(MMET.read_text())
    # Normalize categories to determine top severity
    cats = {}
    # example keys others may have set earlier
    for key in ("var","crps","psi","freshness","manifest","health"):
        if key in m:
            v = m[key]
            if isinstance(v, dict) and "status" in v:
                cats[key] = v["status"]
    # manifest explicitly
    if "manifest" in m and isinstance(m["manifest"], dict):
        cats["manifest"] = m["manifest"].get("status","UNKNOWN")

    top_cat = "OK"
    if cats:
        top_cat = max(cats.values(), key=severity_rank)
    return m, cats, top_cat

def route_message(sev, title, body, cfg):
    routes = (cfg.get("routing") or {}).get(sev) or ["console"]
    sent = []
    if "console" in routes:
        print(f"[{sev}] {title}\n{body}")
        sent.append("console")

    # Slack (optional)
    if "slack" in routes:
        url = cfg.get("slack_webhook")
        if url:
            try:
                import requests
                payload = {"text": f"[{sev}] {title}\n{body}"}
                r = requests.post(url, json=payload, timeout=5)
                if r.status_code//100 == 2:
                    sent.append("slack")
                else:
                    print(f"[alerts] slack post failed: {r.status_code}", file=sys.stderr)
            except Exception as e:
                print(f"[alerts] slack error: {e}", file=sys.stderr)
        else:
            print("[alerts] slack route configured but no webhook provided", file=sys.stderr)

    # (Optional) email route placeholder
    if "email" in routes:
        # integrate with your MTA/API if desired
        print("[alerts] email route configured (placeholder)", file=sys.stderr)
        sent.append("email")

    return sent

def main():
    ap = argparse.ArgumentParser(description="Emit alerts with severity tags/routing.")
    ap.add_argument("--dry", action="store_true", help="Print only; do not send to external routes.")
    args = ap.parse_args()

    cfg = {}
    if CFG.exists():
        try:
            cfg = json.loads(CFG.read_text())
        except Exception:
            pass

    m, cats, top = gather_metrics()

    title = "CL_Model Daily Monitor"
    # Build a compact body
    lines = []
    # Key sections if present
    if "var" in m:
        var = m["var"]
        lines.append(f"VaR5={var.get('var5_exceed')} / VaR95={var.get('var95_exceed')} (n={var.get('n')}) status={var.get('status','UNKNOWN')}")
    if "crps" in m:
        lines.append(f"CRPS_mean={m['crps'].get('mean'):.6f}" if isinstance(m['crps'].get('mean'), (int,float)) else f"CRPS status={m['crps'].get('status','UNKNOWN')}")
    if "manifest" in m:
        man = m["manifest"]
        lines.append(f"Manifest status={man.get('status')} crit={man.get('critical')} warn={man.get('warn')} driftA={man.get('drift_alerts')} driftW={man.get('drift_warns')}")
    if "health" in m:
        h = m["health"]
        net = h.get("net",{})
        lines.append(f"Health net_ok={net.get('ok')} ms={net.get('ms')} disk_free_gb={h.get('disk',{{}}).get('free_gb')}")

    body = "\n".join(lines) if lines else json.dumps(m, indent=2)

    sev = top
    if args.dry:
        print(f"[{sev}] {title}\n{body}")
        return

    sent = route_message(sev, title, body, cfg)
    print(f"[alerts] sent via: {','.join(sent) if sent else 'none'} (severity={sev})")

if __name__ == "__main__":
    main()
