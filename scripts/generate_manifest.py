import argparse, json, numpy as np, pandas as pd
from pathlib import Path

DEF_FEATURES = Path("/Users/georgekurchey/CL_Model/data/proc/features.csv")
DEF_OUT = Path("/Users/georgekurchey/CL_Model/config/features_manifest.json")

def infer_dtype(s: pd.Series):
    if pd.api.types.is_bool_dtype(s): return "bool"
    if pd.api.types.is_integer_dtype(s): return "int"
    if pd.api.types.is_float_dtype(s): return "float"
    if pd.api.types.is_datetime64_any_dtype(s): return "datetime"
    return "string"

def main():
    ap = argparse.ArgumentParser(description="Create (or refresh) features manifest from features.csv")
    ap.add_argument("--features", default=str(DEF_FEATURES))
    ap.add_argument("--out", default=str(DEF_OUT))
    ap.add_argument("--q_low", type=float, default=0.001)
    ap.add_argument("--q_high", type=float, default=0.999)
    ap.add_argument("--nan_max_default", type=float, default=0.10)
    ap.add_argument("--psi_bins", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_csv(args.features, parse_dates=["date"]).sort_values("date")
    mf = {"generated_at": pd.Timestamp.utcnow().isoformat(),
          "rows": int(len(df)),
          "columns": {},
          "drift": {"recent_days": 60, "z_warn": 2.0, "z_alert": 3.0},
          "psi": {"bins": int(args.psi_bins), "warn": 0.10, "alert": 0.25}}

    for c in df.columns:
        if c == "date":
            mf["columns"][c] = {"dtype": "datetime", "required": True}
            continue
        s = df[c]
        # dtype + NaN
        def_dtype = infer_dtype(s)
        rec = {"dtype": def_dtype, "required": True}
        miss = float(s.isna().mean())
        rec["nan_share_baseline"] = miss
        rec["nan_max"] = max(miss, args.nan_max_default)

        if def_dtype in ("float","int","bool"):
            x = pd.to_numeric(s, errors="coerce")
            if x.notna().any():
                rec["mean"] = float(x.mean())
                rec["std"]  = float(x.std(ddof=0))
                ql = float(x.quantile(args.q_low)); qh = float(x.quantile(args.q_high))
                if ql <= qh: rec["allowed_range"] = [ql, qh]
                # PSI edges
                qs = np.linspace(0,1,args.psi_bins+1)
                edges = list(map(float, np.quantile(x.dropna().values, qs)))
                for i in range(1,len(edges)):
                    if edges[i] <= edges[i-1]:
                        edges[i] = edges[i-1] + 1e-12
                rec["psi_edges"] = edges
        mf["columns"][c] = rec

    Path(args.out).write_text(json.dumps(mf, indent=2))
    print(f"[manifest] -> {args.out} (cols={len(mf['columns'])})")

if __name__ == "__main__":
    main()
