import argparse, json, math, numpy as np, pandas as pd
from pathlib import Path

DEF_MANIFEST = Path("/Users/georgekurchey/CL_Model/config/features_manifest.json")
DEF_FEATURES = Path("/Users/georgekurchey/CL_Model/data/proc/features.csv")
DEF_JSON = Path("/Users/georgekurchey/CL_Model/reports/feature_validation.json")
DEF_HTML = Path("/Users/georgekurchey/CL_Model/reports/feature_validation.html")

def psi_from_edges(cur, edges):
    cur = pd.Series(cur).replace([np.inf,-np.inf], np.nan).dropna().values
    if len(cur) < 10 or len(edges) < 3: return float("nan")
    a_hist = np.histogram(cur, bins=np.array(edges, float))[0].astype(float)
    a_pct = np.maximum(a_hist / max(1,a_hist.sum()), 1e-6)
    e_pct = np.full_like(a_pct, 1.0/len(a_pct))
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))

def main():
    ap = argparse.ArgumentParser(description="Validate features: presence/dtype, NaN thresholds, drift checks.")
    ap.add_argument("--manifest", default=str(DEF_MANIFEST))
    ap.add_argument("--features", default=str(DEF_FEATURES))
    ap.add_argument("--out_json", default=str(DEF_JSON))
    ap.add_argument("--out_html", default=str(DEF_HTML))
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    mf = json.loads(Path(args.manifest).read_text())
    df = pd.read_csv(args.features, parse_dates=["date"]).sort_values("date")
    recent_days = int(mf.get("drift",{}).get("recent_days", 60))
    z_warn = float(mf.get("drift",{}).get("z_warn", 2.0))
    z_alert = float(mf.get("drift",{}).get("z_alert", 3.0))
    recent = df.tail(recent_days) if len(df) >= recent_days else df.copy()

    checks, drift_rows = [], []
    critical, warn = False, False

    # Presence / dtype / NaN / range
    for col, spec in mf["columns"].items():
        req = spec.get("required", True)
        if col not in df.columns:
            lvl = "CRITICAL" if req else "WARN"
            checks.append({"column": col, "check": "presence", "status": lvl, "detail": "missing"})
            critical = critical or (lvl=="CRITICAL"); warn = warn or (lvl=="WARN")
            continue

        s = df[col]
        dtype = spec.get("dtype","float")
        # dtype coercion check
        ok_dtype = True
        try:
            if dtype in ("float","int","bool"): pd.to_numeric(s, errors="coerce")
            elif dtype == "datetime": pd.to_datetime(s, errors="coerce")
            else: s.astype(str)
        except Exception as e:
            ok_dtype = False
            checks.append({"column": col, "check": "dtype", "status": "CRITICAL", "detail": f"coercion failed: {e}"})
            critical = True
        if ok_dtype:
            checks.append({"column": col, "check": "dtype", "status": "OK", "detail": dtype})

        nan_share = float(s.isna().mean()); nan_max = float(spec.get("nan_max", 0.10))
        if nan_share > nan_max:
            checks.append({"column": col, "check": "nan_share", "status": "CRITICAL", "detail": f"{nan_share:.3f} > {nan_max:.3f}"})
            critical = True
        else:
            checks.append({"column": col, "check": "nan_share", "status": "OK", "detail": f"{nan_share:.3f} <= {nan_max:.3f}"})

        rng = spec.get("allowed_range")
        if dtype in ("float","int") and rng:
            lo, hi = rng
            x = pd.to_numeric(s, errors="coerce")
            out_pct = float(((x < lo) | (x > hi)).mean())
            lvl = "WARN" if out_pct > 0.01 else "OK"
            if lvl == "WARN": warn = True
            checks.append({"column": col, "check": "range", "status": lvl, "detail": f"{out_pct:.3f} outside [{lo:.4g},{hi:.4g}]"})
        else:
            checks.append({"column": col, "check": "range", "status": "OK", "detail": "n/a"})

    # Drift: recent-vs-historical z-score + PSI (if edges present)
    for col, spec in mf["columns"].items():
        if spec.get("dtype") not in ("float","int"): 
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        xr = pd.to_numeric(recent[col], errors="coerce")
        xb = x.iloc[:-recent_days] if len(x) > recent_days else x
        if xb.dropna().size < 30 or xr.dropna().size < 10: 
            continue
        mu_b, sd_b = float(xb.mean()), float(xb.std(ddof=0))
        mu_r = float(xr.mean())
        if sd_b > 0:
            z = (mu_r - mu_b) / sd_b
            lvl = "OK"
            if abs(z) >= z_alert: lvl, warn = "ALERT", True
            elif abs(z) >= z_warn: lvl, warn = "WARN", True
        else:
            z, lvl = float("nan"), "OK"

        psi_score = float("nan")
        if "psi_edges" in spec:
            psi_score = psi_from_edges(xr.values, spec["psi_edges"])
        drift_rows.append({"column": col,
                           "z": None if math.isnan(z) else round(float(z),3),
                           "psi": None if (isinstance(psi_score,float) and math.isnan(psi_score)) else round(float(psi_score),3),
                           "level": lvl})

    rep = {"asof": pd.Timestamp.utcnow().isoformat(),
           "rows": int(len(df)), "recent_days": int(recent_days),
           "critical": critical, "warn": warn,
           "checks": checks, "drift": drift_rows}
    Path(DEF_JSON).write_text(json.dumps(rep, indent=2))

    html = ["<html><head><meta charset='utf-8'><title>Feature Validation</title>",
            "<style>body{font-family:Arial,Helvetica,sans-serif;margin:20px;} table{border-collapse:collapse;} th,td{border:1px solid #ddd;padding:6px;} th{background:#f7f7f7;}</style>",
            "</head><body>",
            f"<h2>CL_Model — Feature Validation</h2><p>asof: {rep['asof']} | rows: {rep['rows']} | recent_days: {rep['recent_days']}</p>",
            pd.DataFrame(checks).to_html(index=False)]
    if drift_rows: html.append(pd.DataFrame(drift_rows).to_html(index=False))
    html.append(f"<p>Status: <b>{{'CRITICAL' if critical else ('WARN' if warn else 'OK')}}</b></p>")
    html.append("</body></html>")
    Path(DEF_HTML).write_text("\n".join(html), encoding="utf-8")

    if args.strict and critical:
        raise SystemExit(2)

    print(f"[validate] json -> {DEF_JSON}")
    print(f"[validate] html -> {DEF_HTML}")
    print(f"[validate] status -> {{'CRITICAL' if critical else ('WARN' if warn else 'OK')}}")

if __name__ == "__main__":
    main()
