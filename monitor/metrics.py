import argparse, json, math, traceback
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/georgekurchey/CL_Model")
DEF_CONFIG = ROOT / "config/monitor.json"
DEF_FEATURES = ROOT / "data/proc/features.csv"
DEF_LIVE = ROOT / "preds/live_history.csv"
DEF_OUT = ROOT / "reports/monitor_metrics.json"

def wilson_interval(k, n, z=1.96):
    if n <= 0: return (float("nan"), float("nan"), float("nan"))
    p_hat = k / n
    denom = 1 + z*z/n
    center = (p_hat + z*z/(2*n)) / denom
    half = (z/denom) * math.sqrt((p_hat*(1-p_hat)/n) + (z*z/(4*n*n)))
    return (p_hat, max(0.0, center - half), min(1.0, center + half))

def crps_from_grid(y, q_grid, taus):
    taus = np.array(taus, dtype=float)
    if q_grid.size == 0 or len(taus) == 0: return np.zeros(len(y))
    w = np.empty_like(taus)
    w[0] = (taus[1] - 0.0)/2 if len(taus) > 1 else 1.0
    w[-1] = (1.0 - taus[-2])/2 if len(taus) > 1 else 1.0
    if len(taus) > 2:
        w[1:-1] = (taus[2:] - taus[:-2]) / 2
    L = []
    for i,t in enumerate(taus):
        e = y - q_grid[:,i]
        L.append(np.maximum(t*e, (t-1)*e))
    L = np.vstack(L).T if L else np.zeros((len(y),1))
    return (L*w).sum(axis=1) if L.ndim == 2 else np.zeros(len(y))

def psi(actual: np.ndarray, expected: np.ndarray, bins: int = 10):
    a = pd.Series(actual).replace([np.inf,-np.inf], np.nan).dropna()
    e = pd.Series(expected).replace([np.inf,-np.inf], np.nan).dropna()
    if len(a) < 5 or len(e) < 50: return float("nan")
    qs = np.linspace(0, 1, bins+1)
    edges = np.quantile(e, qs)
    edges = np.unique(edges)
    if len(edges) < 3: return float("nan")
    a_hist = np.histogram(a, bins=edges)[0].astype(float)
    e_hist = np.histogram(e, bins=edges)[0].astype(float)
    a_pct = np.maximum(a_hist / max(1,a_hist.sum()), 1e-6)
    e_pct = np.maximum(e_hist / max(1,e_hist.sum()), 1e-6)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))

def safe_latest_date(path: Path):
    try:
        if not path.exists(): return None
        if path.suffix.lower() in [".parquet", ".pq"]:
            # Parquet requires pyarrow/fastparquet; guard it
            try:
                df = pd.read_parquet(path)
            except Exception:
                return None
        else:
            # try CSV with 'date' column
            try:
                hdr = pd.read_csv(path, nrows=0)
                if "date" in hdr.columns:
                    df = pd.read_csv(path, parse_dates=["date"])
                else:
                    return None
            except Exception:
                return None
        if "date" in df.columns:
            return pd.to_datetime(df["date"], errors="coerce").max()
        return None
    except Exception:
        return None

def compute(args):
    cfg = json.loads(Path(args.config).read_text())

    feats = pd.read_csv(args.features, parse_dates=["date"]).sort_values("date")
    feats["date"] = pd.to_datetime(feats["date"]).dt.tz_localize(None)

    live = pd.DataFrame(columns=["date"])
    live_path = Path(args.live)
    if live_path.exists():
        live = pd.read_csv(live_path, parse_dates=["date"]).sort_values("date")
        live["date"] = pd.to_datetime(live["date"]).dt.tz_localize(None)

    df = live.merge(feats[["date","ret_1d"]], on="date", how="left").rename(columns={"ret_1d":"y"})

    tau_cols = sorted([c for c in df.columns if c.startswith("q") and len(c)==3 and c[1:].isdigit()],
                      key=lambda c: int(c[1:]))
    taus = [int(c[1:])/100.0 for c in tau_cols] if tau_cols else []

    df_av = df.loc[df["y"].notna()].copy()
    n = len(df_av)

    if n > 0 and "q05" in df_av and "q95" in df_av:
        y = df_av["y"].values
        q05 = df_av["q05"].values
        q95 = df_av["q95"].values
        k5 = int((y < q05).sum()); k95 = int((y > q95).sum())
        p5, lo5, hi5 = wilson_interval(k5, n, z=cfg.get("var_z",1.96))
        p95, lo95, hi95 = wilson_interval(k95, n, z=cfg.get("var_z",1.96))
        q_grid = df_av[tau_cols].values if tau_cols else np.empty((n,0))
        crps = crps_from_grid(y, q_grid, taus) if (n>0 and len(taus)>0) else np.array([])
        wins = cfg["windows"]
        crps_short = float(np.mean(crps[-wins["short"]:])) if len(crps) >= wins["short"] else float("nan")
        base_start = max(0, len(crps)-wins["long"]-wins["short"])
        base_end = max(0, len(crps)-wins["short"])
        crps_base = float(np.mean(crps[base_start:base_end])) if base_end>base_start else float("nan")
        crps_drift = ((crps_short - crps_base) / crps_base) if (not math.isnan(crps_short) and not math.isnan(crps_base) and crps_base!=0.0) else float("nan")
    else:
        k5=k95=0; p5=lo5=hi5=p95=lo95=hi95=float("nan")
        wins = cfg["windows"]; crps_short=crps_base=crps_drift=float("nan")

    numeric_cols = [c for c in feats.columns if c not in ("date","ret_1d") and np.issubdtype(feats[c].dtype, np.number)]
    psi_map = {}
    if len(feats) > 400:
        ref = feats.iloc[-(wins["long"]+252):-wins["long"]][numeric_cols] if len(feats) > (wins["long"]+252) else feats.iloc[:-wins["long"]][numeric_cols]
        cur = feats.iloc[-wins["short"]:,][numeric_cols]
        for c in numeric_cols:
            psi_map[c] = psi(cur[c].values, ref[c].values)
    else:
        psi_map = {c: float("nan") for c in numeric_cols}

    latest = {
        "features": feats["date"].max(),
        "live_pred": live["date"].max() if not live.empty else None,
        "cl_strip": safe_latest_date(ROOT/"data/raw/cl_strip.parquet"),
        "macro": safe_latest_date(ROOT/"data/raw/macro.parquet"),
        "eia": safe_latest_date(ROOT/"data/raw/eia.parquet")
    }
    latest_str = {k: (str(v.date()) if (v is not None and not pd.isna(v)) else None) for k,v in latest.items()}

    out = {
        "asof": str(pd.Timestamp.utcnow()),
        "counts": {"n_realized": int(n), "notes": ("no_live_history" if live.empty else None)},
        "coverage_total": {"n": int(n), "var5": {"k": int(k5), "p": p5, "lo": lo5, "hi": hi5}, "var95": {"k": int(k95), "p": p95, "lo": lo95, "hi": hi95}},
        "coverage_roll": {"short": None, "medium": None, "long": None},
        "crps": {"short_mean": crps_short, "baseline_mean": crps_base, "pct_change": crps_drift},
        "psi": psi_map,
        "latest_dates": latest_str
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[metrics] -> {args.out}")

def main():
    ap = argparse.ArgumentParser(description="Compute monitoring metrics (VaR coverage, CRPS, PSI, freshness).")
    ap.add_argument("--config", default=str(DEF_CONFIG))
    ap.add_argument("--features", default=str(DEF_FEATURES))
    ap.add_argument("--live", default=str(DEF_LIVE))
    ap.add_argument("--out", default=str(DEF_OUT))
    args = ap.parse_args()

    try:
        compute(args)
    except Exception as e:
        # Always write a minimal file so downstream can run
        msg = f"metrics_failed: {e}"
        fallback = {
            "asof": str(pd.Timestamp.utcnow()),
            "counts": {"n_realized": 0, "notes": msg},
            "coverage_total": {"n": 0, "var5": {"k": 0, "p": float("nan"), "lo": float("nan"), "hi": float("nan")},
                               "var95": {"k": 0, "p": float("nan"), "lo": float("nan"), "hi": float("nan")}},
            "coverage_roll": {"short": None, "medium": None, "long": None},
            "crps": {"short_mean": float("nan"), "baseline_mean": float("nan"), "pct_change": float("nan")},
            "psi": {},
            "latest_dates": {}
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(fallback, indent=2))
        print("[metrics] WROTE FALLBACK due to error.")
        traceback.print_exc()
if __name__ == "__main__":
    main()
