
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

DEF_FEATURES = Path("/Users/georgekurchey/CL_Model/data/proc/features.csv")
DEF_OUT = Path("/Users/georgekurchey/CL_Model/config/features_manifest.json")

def infer_dtype(series: pd.Series):
    if pd.api.types.is_bool_dtype(series): return "bool"
    if pd.api.types.is_integer_dtype(series): return "int"
    if pd.api.types.is_float_dtype(series): return "float"
    if pd.api.types.is_datetime64_any_dtype(series): return "datetime"
    return "string"

def main():
    ap = argparse.ArgumentParser(description="Bootstrap features manifest from a features.csv snapshot.")
    ap.add_argument("--features", default=str(DEF_FEATURES))
    ap.add_argument("--out", default=str(DEF_OUT))
    ap.add_argument("--psi_bins", type=int, default=10)
    ap.add_argument("--nan_max_default", type=float, default=0.10, help="Default max NaN share per column.")
    ap.add_argument("--q_low", type=float, default=0.001, help="Lower quantile for allowed range.")
    ap.add_argument("--q_high", type=float, default=0.999, help="Upper quantile for allowed range.")
    args = ap.parse_args()

    df = pd.read_csv(args.features, parse_dates=["date"]).sort_values("date")
    manifest = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "rows": int(len(df)),
        "columns": {},
        "psi": { "bins": int(args.psi_bins), "warn": 0.10, "alert": 0.25 },
        "notes": "Auto-generated baseline from current features.csv. Update if schema changes."
    }

    for col in df.columns:
        if col == "date":  # keep as datetime presence only
            manifest["columns"][col] = { "dtype": "datetime", "required": True }
            continue
        s = df[col]
        dtype = infer_dtype(s)
        rec = { "dtype": dtype, "required": True }
        miss = float(s.isna().mean())
        rec["nan_share_baseline"] = miss
        rec["nan_max"] = max(miss, args.nan_max_default)

        if dtype in ("float","int","bool"):
            s_num = pd.to_numeric(s, errors="coerce")
            rec["mean"] = float(s_num.mean(skipna=True)) if s_num.notna().any() else None
            rec["std"] = float(s_num.std(skipna=True, ddof=0)) if s_num.notna().any() else None
            ql = float(s_num.quantile(args.q_low)) if s_num.notna().any() else None
            qh = float(s_num.quantile(args.q_high)) if s_num.notna().any() else None
            if ql is not None and qh is not None and ql <= qh:
                rec["allowed_range"] = [ql, qh]
            # store decile edges for PSI reference
            if s_num.notna().sum() >= 100:
                qs = np.linspace(0,1,int(args.psi_bins)+1)
                edges = list(map(float, np.quantile(s_num.dropna().values, qs)))
                # make strictly increasing to avoid zero-width bins
                for i in range(1,len(edges)):
                    if edges[i] <= edges[i-1]:
                        edges[i] = edges[i-1] + 1e-12
                rec["psi_edges"] = edges
        else:
            # string/type columns not expected, but record basic info
            rec["categories"] = None

        manifest["columns"][col] = rec

    Path(args.out).write_text(json.dumps(manifest, indent=2))
    print(f"[manifest] -> {args.out} (cols={len(manifest['columns'])})")

if __name__ == "__main__":
    main()
