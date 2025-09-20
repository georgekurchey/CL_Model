import json
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/georgekurchey/CL_Model")
FVAL = ROOT/"reports/feature_validation.json"
MMET = ROOT/"reports/monitor_metrics.json"

def main():
    # load existing metrics if present
    base = {}
    if MMET.exists():
        try:
            base = json.loads(MMET.read_text())
        except Exception:
            base = {}
    # load feature validation
    if not FVAL.exists():
        # write a soft marker so alerts can still run
        base["manifest"] = {"status":"UNKNOWN","critical":None,"warn":None,"missing_cols":[], "drift_alerts":None, "drift_warns":None}
        MMET.write_text(json.dumps(base, indent=2))
        print("[metrics] manifest: feature_validation.json not found (status=UNKNOWN)")
        return

    rep = json.loads(FVAL.read_text())
    checks = rep.get("checks", [])
    drift = rep.get("drift", [])

    critical = sum(1 for c in checks if c.get("status")=="CRITICAL")
    warns    = sum(1 for c in checks if c.get("status")=="WARN")
    missing  = [c.get("column") for c in checks if c.get("check")=="presence" and c.get("status") in ("CRITICAL","WARN")]
    d_alerts = sum(1 for d in drift if d.get("level")=="ALERT")
    d_warns  = sum(1 for d in drift if d.get("level")=="WARN")

    status = "OK"
    if critical>0 or d_alerts>0: status = "CRITICAL"
    elif warns>0 or d_warns>0: status = "WARN"

    base["manifest"] = {
        "status": status,
        "critical": critical,
        "warn": warns,
        "missing_cols": missing[:5],
        "drift_alerts": d_alerts,
        "drift_warns": d_warns,
        "asof": pd.Timestamp.utcnow().isoformat()
    }
    MMET.write_text(json.dumps(base, indent=2))
    print(f"[metrics] manifest status={status} critical={critical} warn={warns} drift_alerts={d_alerts} drift_warns={d_warns}")

if __name__ == "__main__":
    main()
