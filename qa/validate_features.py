
import argparse, json, math
from pathlib import Path
import numpy as np, pandas as pd

DEF_MANIFEST = Path("/Users/georgekurchey/CL_Model/config/features_manifest.json")
DEF_FEATURES = Path("/Users/georgekurchey/CL_Model/data/proc/features.csv")
DEF_OUT_JSON = Path("/Users/georgekurchey/CL_Model/reports/feature_validation.json")
DEF_OUT_HTML = Path("/Users/georgekurchey/CL_Model/reports/feature_validation.html")

def psi_from_edges(cur: np.ndarray, edges: np.ndarray):
    # robust PSI on numeric with provided bin edges
    cur = pd.Series(cur).replace([np.inf,-np.inf], np.nan).dropna().values
    if len(cur) < 10 or len(edges) < 3: return float("nan")
    # expected: uniform over bins (baseline unknown proportions), acceptable for guardrail
    a_hist = np.histogram(cur, bins=edges)[0].astype(float)
    a_pct = np.maximum(a_hist / max(1,a_hist.sum()), 1e-6)
    e_pct = np.full_like(a_pct, 1.0/len(a_pct))
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))

def html_table(df: pd.DataFrame, title: str):
    return f"<h3>{title}</h3>" + df.to_html(index=False)

def main():
    ap = argparse.ArgumentParser(description="Validate features against manifest (schema, NaNs, ranges, drift).")
    ap.add_argument("--manifest", default=str(DEF_MANIFEST))
    ap.add_argument("--features", default=str(DEF_FEATURES))
    ap.add_argument("--out_json", default=str(DEF_OUT_JSON))
    ap.add_argument("--out_html", default=str(DEF_OUT_HTML))
    ap.add_argument("--recent_days", type=int, default=60, help="Window for drift checks.")
    ap.add_argument("--strict", action="store_true", help="Exit non-zero on any CRITICAL error.")
    args = ap.parse_args()

    mf = json.loads(Path(args.manifest).read_text())
    df = pd.read_csv(args.features, parse_dates=["date"]).sort_values("date")
    recent = df.tail(args.recent_days).copy() if len(df) >= args.recent_days else df.copy()

    checks = []
    critical = False
    warn = False

    # schema + dtype + NaN + range
    for col, spec in mf["columns"].items():
        required = spec.get("required", True)
        present = col in df.columns
        status = "OK"
        msg = ""
        if not present:
            if required:
                status, msg, critical = "CRITICAL", "missing required column", True
            else:
                status, msg, warn = "WARN", "missing optional column", True
            checks.append({"column": col, "check": "presence", "status": status, "detail": msg})
            continue

        # dtype (coercible)
        dtype = spec.get("dtype","float")
        s = df[col]
        try:
            if dtype in ("float","int","bool"):
                s_num = pd.to_numeric(s, errors="coerce")
            elif dtype == "datetime":
                pd.to_datetime(s, errors="coerce")
            else:
                s.astype(str)
        except Exception as e:
            checks.append({"column": col, "check": "dtype", "status": "CRITICAL", "detail": f"type coercion failed: {e}"})
            critical = True
            continue

        # NaN share
        nan_share = float(s.isna().mean())
        nan_max = float(spec.get("nan_max", 0.10))
        if nan_share > nan_max:
            checks.append({"column": col, "check": "nan_share", "status": "CRITICAL", "detail": f"{nan_share:.3f} > {nan_max:.3f}"})
            critical = True
        else:
            checks.append({"column": col, "check": "nan_share", "status": "OK", "detail": f"{nan_share:.3f} <= {nan_max:.3f}"})

        # range
        rng = spec.get("allowed_range")
        if dtype in ("float","int") and rng:
            lo, hi = rng
            s_num = pd.to_numeric(s, errors="ignore")
            out_pct = float(((s_num < lo) | (s_num > hi)).mean())
            if out_pct > 0.01:  # >1% outside baseline extreme range
                checks.append({"column": col, "check": "range", "status": "WARN", "detail": f"{out_pct:.3f} outside [{lo:.4g},{hi:.4g}] (baseline extremes)"})
                warn = True
            else:
                checks.append({"column": col, "check": "range", "status": "OK", "detail": f"{out_pct:.3f} outside allowed"})
        else:
            checks.append({"column": col, "check": "range", "status": "OK", "detail": "n/a"})

    # drift (PSI) on recent window using stored edges (if any)
    psi_warn = float(mf.get("psi",{}).get("warn", 0.10))
    psi_alert = float(mf.get("psi",{}).get("alert", 0.25))
    drift_rows = []
    for col, spec in mf["columns"].items():
        if spec.get("dtype") not in ("float","int"): 
            continue
        edges = spec.get("psi_edges")
        if not edges: 
            continue
        try:
            cur = pd.to_numeric(recent[col], errors="coerce").dropna().values
        except Exception:
            continue
        score = psi_from_edges(cur, np.array(edges, float))
        lvl = "OK"
        if isinstance(score, float) and not math.isnan(score):
            if score >= psi_alert:
                lvl, warn = "ALERT", True
            elif score >= psi_warn:
                lvl, warn = "WARN", True
        drift_rows.append({"column": col, "psi": (None if math.isnan(score) else round(score,4)), "level": lvl})

    # assemble report
    rep = {
        "asof": pd.Timestamp.utcnow().isoformat(),
        "rows": int(len(df)),
        "recent_days": int(args.recent_days),
        "critical": critical,
        "warn": warn,
        "checks": checks,
        "drift": drift_rows
    }
    Path(args.out_json).write_text(json.dumps(rep, indent=2))

    # HTML
    html = ["<html><head><meta charset='utf-8'><title>Feature Validation</title>",
            "<style>body{font-family:Arial,Helvetica,sans-serif;margin:20px;} table{border-collapse:collapse;} th,td{border:1px solid #ddd;padding:6px;} th{background:#f7f7f7;}</style>",
            "</head><body>",
            "<h2>CL_Model — Feature Validation</h2>",
            f"<p>asof: {rep['asof']} | rows: {rep['rows']} | recent_days: {rep['recent_days']}</p>"]
    df_checks = pd.DataFrame(checks)
    if not df_checks.empty:
        html.append(df_checks.to_html(index=False))
    else:
        html.append("<p>No checks recorded.</p>")
    if drift_rows:
        html.append(html_table(pd.DataFrame(drift_rows), "Drift (PSI vs. baseline edges)"))
    html.append(f"<p>Status: <b>{'CRITICAL' if critical else ('WARN' if warn else 'OK')}</b></p>")
    html.append("</body></html>")
    Path(args.out_html).write_text("\n".join(html), encoding="utf-8")

    # exit code
    if args.strict and critical:
        raise SystemExit(2)

    print(f"[validate] json -> {args.out_json}")
    print(f"[validate] html -> {args.out_html}")
    print(f"[validate] status -> {'CRITICAL' if critical else ('WARN' if warn else 'OK')}")

if __name__ == "__main__":
    main()
