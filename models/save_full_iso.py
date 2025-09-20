import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
import joblib

DEF_ROOT = Path("/Users/georgekurchey/CL_Model")
DEF_FEATURES = DEF_ROOT / "data/proc/features.csv"
DEF_MODELS = DEF_ROOT / "models" / "gbt_iso.joblib"

def parse_taus(s: str):
    parts = sorted(set(float(x) for x in s.split(",") if x.strip()))
    assert all(0.0 < t < 1.0 for t in parts)
    return parts

def load_features(path: Path):
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date").reset_index(drop=True)
    if "ret_1d" not in df.columns:
        raise SystemExit("ret_1d missing in features.")
    return df

def build_xy(df: pd.DataFrame):
    y = df["ret_1d"].astype(float).values
    X = df.drop(columns=["ret_1d","date"], errors="ignore")
    num = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
    return X[num].copy(), y, num

def fit_grid(Xtr, ytr, taus, params):
    models = {}
    for t in taus:
        m = GradientBoostingRegressor(loss="quantile",
                                      alpha=t,
                                      n_estimators=params["n_estimators"],
                                      learning_rate=params["learning_rate"],
                                      max_depth=params["max_depth"],
                                      min_samples_leaf=params["min_samples_leaf"],
                                      random_state=params["random_state"])
        m.fit(Xtr, ytr)
        models[t] = m
    return models

def iso_inverse(taus, y_cal, q_cal):
    taus = np.asarray(taus, float)
    cover = np.array([np.mean(y_cal <= q_cal[:,j]) for j,_ in enumerate(taus)])
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    fitted = iso.fit_transform(taus, cover)  # f(tau) ~ coverage
    xs = fitted.copy(); ys = taus.copy()
    for i in range(1, len(xs)):
        if xs[i] <= xs[i-1]: xs[i] = xs[i-1] + 1e-6
    def inv_f(t_star: float) -> float:
        t_star = float(np.clip(t_star, 0.0, 1.0))
        return float(np.interp(t_star, xs, ys))
    return inv_f, {"taus":taus.tolist(),"cover":cover.tolist(),"fitted":fitted.tolist()}

def main():
    ap = argparse.ArgumentParser(description="Train full GBT-ISO model (retrain if stale).")
    ap.add_argument("--features", default=str(DEF_FEATURES))
    ap.add_argument("--out", default=str(DEF_MODELS))
    ap.add_argument("--iso_days", type=int, default=252)
    ap.add_argument("--taus", type=str, default="0.05,0.15,0.25,0.35,0.50,0.65,0.75,0.85,0.95")
    ap.add_argument("--n_estimators", type=int, default=600)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--max_depth", type=int, default=3)
    ap.add_argument("--min_samples_leaf", type=int, default=60)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--refresh_days", type=int, default=30, help="retrain if model older than this")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    outp = Path(args.out)
    if outp.exists() and not args.force:
        age_days = (time.time() - outp.stat().st_mtime)/86400.0
        if age_days < args.refresh_days:
            print(f"[save_full_iso] Existing model is fresh ({age_days:.1f}d). SKIP retrain.")
            return

    df = load_features(Path(args.features))
    X, y, cols = build_xy(df)
    if len(df) <= args.iso_days + 250:
        raise SystemExit("Not enough rows for core+calibration. Reduce iso_days.")

    core_end = len(df) - args.iso_days
    Xcore, ycore = X.iloc[:core_end].values, y[:core_end]
    Xcal, ycal = X.iloc[core_end:].values, y[core_end:]

    imp = SimpleImputer(strategy="median")
    Xcore = imp.fit_transform(Xcore)
    Xcal  = imp.transform(Xcal)

    taus = parse_taus(args.taus)
    params = dict(n_estimators=args.n_estimators, learning_rate=args.learning_rate,
                  max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf,
                  random_state=args.random_state)
    models = fit_grid(Xcore, ycore, taus, params)

    q_cal = np.column_stack([models[t].predict(Xcal) for t in taus])
    q_cal = np.sort(q_cal, axis=1)
    inv_f, iso_diag = iso_inverse(taus, ycal, q_cal)

    t_eff = {"0.05": inv_f(0.05), "0.50": inv_f(0.50), "0.95": inv_f(0.95)}
    artifact = {
        "taus": taus,
        "cols": cols,
        "imputer": imp,
        "models": models,
        "t_eff": t_eff,
        "iso_diag": iso_diag,
        "meta": {"trained_at": pd.Timestamp.utcnow().isoformat(), "iso_days": args.iso_days, "params": params}
    }
    outp.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, outp)
    print(f"[save_full_iso] Saved -> {outp}")
if __name__ == "__main__":
    main()
