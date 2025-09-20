import json, shutil, sys, time
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/georgekurchey/CL_Model")
OUT = ROOT / "reports/health.json"

def _latency():
    try:
        import requests
    except Exception:
        return {"ok": False, "ms": None, "endpoint": None, "err": "requests_not_installed"}
    endpoints = ["https://www.google.com/generate_204", "https://api.github.com/"]
    last_err = None
    for ep in endpoints:
        t0 = time.perf_counter()
        try:
            r = requests.get(ep, timeout=3)
            dt = int((time.perf_counter()-t0)*1000)
            if r.status_code in (204, 200):
                return {"ok": True, "ms": dt, "endpoint": ep, "err": None}
        except Exception as e:
            last_err = str(e)
    return {"ok": False, "ms": None, "endpoint": endpoints[-1], "err": last_err}

def _file_info(p: Path, date_col="date"):
    info = {"exists": p.exists(), "rows": None, "mb": None, "mtime": None, "latest_date": None}
    if not p.exists(): return info
    try:
        info["mb"] = round(p.stat().st_size / (1024*1024), 3)
        info["mtime"] = pd.Timestamp(p.stat().st_mtime, unit="s").isoformat()
        if p.suffix.lower() == ".csv":
            # light read; count rows quickly
            info["rows"] = sum(1 for _ in open(p, "r", encoding="utf-8", errors="ignore")) - 1
            df = pd.read_csv(p, nrows=2000)
            if date_col in df.columns:
                ld = pd.to_datetime(df[date_col], errors="coerce").max()
                if pd.notna(ld):
                    info["latest_date"] = pd.to_datetime(ld).tz_localize(None).isoformat()
    except Exception:
        pass
    return info

def main():
    py = {"executable": sys.executable, "version": sys.version.split()[0]}
    try:
        import platform
        py["platform"] = platform.platform()
    except Exception:
        pass

    du = shutil.disk_usage("/")
    disk = {"free_gb": round(du.free/1e9, 2), "total_gb": round(du.total/1e9, 2)}

    net = _latency()

    feats = _file_info(ROOT/"data/proc/features.csv")
    live = _file_info(ROOT/"preds/live_today.csv")
    metrics_p = ROOT/"reports/monitor_metrics.json"
    metrics = {"exists": metrics_p.exists(),
               "mb": round(metrics_p.stat().st_size/(1024*1024),3) if metrics_p.exists() else None}

    out = {
        "asof": pd.Timestamp.utcnow().isoformat(),
        "python": py,
        "disk": disk,
        "net": net,
        "features_csv": feats,
        "live_today": live,
        "monitor_metrics": metrics
    }
    OUT.write_text(json.dumps(out, indent=2))
    status = "OK" if net.get("ok") else "NET_FAIL"
    lat = net.get("ms")
    feat_rows = feats.get("rows")
    print(f"[health] {status} net_ms={lat} features_rows={feat_rows} disk_free_gb={disk['free_gb']} live_exists={live['exists']} metrics_exists={metrics['exists']}")

if __name__ == "__main__":
    main()
