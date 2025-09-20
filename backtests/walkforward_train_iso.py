import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer

DEF_ROOT = Path("/Users/georgekurchey/CL_Model")
DEF_FEATURES = DEF_ROOT / "data/proc/features.csv"
DEF_PREDS = DEF_ROOT / "preds"

def parse_taus(s: str):
    parts = [float(x) for x in s.split(",") if x.strip()!=""]
    parts = sorted(set(parts))
    assert all(0.0 < t < 1.0 for t in parts), "taus must be in (0,1)"
    return parts

def load_features(path: Path):
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date").reset_index(drop=True)
    if "ret_1d" not in df.columns:
        raise SystemExit("ret_1d column missing in features.")
    return df

def build_xy(df: pd.DataFrame):
    y = df["ret_1d"].astype(float).values
    X = df.drop(columns=["ret_1d","date"], errors="ignore")
    num_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
    X = X[num_cols].copy()
    return X, y, num_cols

def fold_splits(dates: pd.Series, train_years: int, test_months: int, min_train_days:int=250):
    train_days = int(train_years * 252)
    test_days = int(test_months * 21)
    n = len(dates)
    starts = []
    t0 = train_days
    while t0 + test_days <= n:
        tr_start = max(0, t0 - train_days)
        tr_end = t0
        te_end = t0 + test_days
        if tr_end - tr_start >= min_train_days:
            starts.append((tr_start, tr_end, tr_end, te_end))
        t0 += test_days
    return starts

def fit_quantile_models(Xtr, ytr, taus, params):
    models = {}
    for t in taus:
        mdl = GradientBoostingRegressor(loss="quantile",
                                        alpha=t,
                                        n_estimators=params["n_estimators"],
                                        learning_rate=params["learning_rate"],
                                        max_depth=params["max_depth"],
                                        min_samples_leaf=params["min_samples_leaf"],
                                        random_state=params["random_state"])
        mdl.fit(Xtr, ytr)
        models[t] = mdl
    return models

def predict_grid(models, X, taus):
    return np.column_stack([models[t].predict(X) for t in taus])

def isotonic_recalibration(taus, y_cal, q_cal):
    taus = np.asarray(taus, dtype=float)
    cover = np.array([np.mean(y_cal <= q_cal[:, j]) for j,_ in enumerate(taus)], dtype=float)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    fitted = iso.fit_transform(taus, cover)  # f(tau) ≈ empirical coverage
    xs, ys = fitted.copy(), taus.copy()
    for i in range(1, len(xs)):
        if xs[i] <= xs[i-1]:
            xs[i] = xs[i-1] + 1e-6
    def inv_f(t_star: float) -> float:
        t_star = float(np.clip(t_star, 0.0, 1.0))
        return float(np.interp(t_star, xs, ys))
    return inv_f, (taus.tolist(), cover.tolist(), fitted.tolist())

def interp_quantile(row_q, taus, t_star):
    return float(np.interp(t_star, taus, row_q))

def main():
    ap = argparse.ArgumentParser(description="Walk-forward train GBT quantiles with isotonic recalibration.")
    ap.add_argument("--features", default=str(DEF_FEATURES))
    ap.add_argument("--outdir", default=str(DEF_PREDS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train_years", type=float, default=3.0)
    ap.add_argument("--test_months", type=float, default=6.0)
    ap.add_argument("--iso_days", type=int, default=252)
    ap.add_argument("--taus", type=str, default="0.05,0.15,0.25,0.35,0.50,0.65,0.75,0.85,0.95")
    ap.add_argument("--n_estimators", type=int, default=600)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--max_depth", type=int, default=3)
    ap.add_argument("--min_samples_leaf", type=int, default=60)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    taus = parse_taus(args.taus)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    df = load_features(Path(args.features))
    X_all, y_all, cols = build_xy(df)
    splits = fold_splits(df["date"], args.train_years, args.test_months)
    if not splits:
        raise SystemExit("Not enough data to form folds. Reduce train_years/test_months.")
    splits = splits[-args.folds:]

    all_rows, fold_meta = [], []
    for k, (tr_start, tr_end, te_start, te_end) in enumerate(splits, start=1):
        ntr = tr_end - tr_start
        ncal = min(args.iso_days, max(1, int(0.2 * ntr)))
        cal_start = tr_end - ncal
        core_start, core_end = tr_start, cal_start

        Xtr_core = X_all.iloc[core_start:core_end].values
        ytr_core = y_all[core_start:core_end]
        Xcal = X_all.iloc[cal_start:tr_end].values
        ycal = y_all[cal_start:tr_end]
        Xte  = X_all.iloc[te_start:te_end].values
        yte  = y_all[te_start:te_end]
        dte  = df["date"].iloc[te_start:te_end].values

        imp = SimpleImputer(strategy="median")
        Xtr_core = imp.fit_transform(Xtr_core)
        Xcal = imp.transform(Xcal)
        Xte = imp.transform(Xte)

        params = dict(n_estimators=args.n_estimators, learning_rate=args.learning_rate,
                      max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf,
                      random_state=args.random_state)
        models = fit_quantile_models(Xtr_core, ytr_core, taus, params)

        q_cal = predict_grid(models, Xcal, taus)
        inv_f, iso_diag = isotonic_recalibration(taus, ycal, q_cal)
        t05_eff, t50_eff, t95_eff = inv_f(0.05), inv_f(0.50), inv_f(0.95)

        q_te = predict_grid(models, Xte, taus)
        q_te = np.sort(q_te, axis=1)
        q05 = np.array([interp_quantile(q_te[i], taus, t05_eff) for i in range(len(q_te))])
        q50 = np.array([interp_quantile(q_te[i], taus, t50_eff) for i in range(len(q_te))])
        q95 = np.array([interp_quantile(q_te[i], taus, t95_eff) for i in range(len(q_te))])

        grid_cols = [f"q{int(t*100):02d}" for t in taus]
        df_grid = pd.DataFrame(q_te, columns=grid_cols)
        out = pd.DataFrame({"date": dte, "y": yte, "q05": q05, "q50": q50, "q95": q95})
        out = pd.concat([out.reset_index(drop=True), df_grid.reset_index(drop=True)], axis=1)
        out["fold"] = k
        all_rows.append(out)

        fold_meta.append({
            "fold": k,
            "cal_range": [str(df['date'].iloc[cal_start].date()), str(df['date'].iloc[tr_end-1].date())],
            "test_range": [str(df['date'].iloc[te_start].date()), str(df['date'].iloc[te_end-1].date())],
            "taus": taus, "iso_diag": iso_diag,
            "t_eff": {"t05": t05_eff, "t50": t50_eff, "t95": t95_eff},
            "features": cols
        })

        (Path(args.outdir) / f"preds_fold{k}.csv").write_text(out.to_csv(index=False))

    big = pd.concat(all_rows, axis=0, ignore_index=True)
    (Path(args.outdir) / "preds_all.csv").write_text(big.to_csv(index=False))

    meta = {"splits": fold_meta,
            "params": {"train_years": args.train_years, "test_months": args.test_months,
                       "iso_days": args.iso_days, "taus": taus,
                       "gbt": {"n_estimators": args.n_estimators, "learning_rate": args.learning_rate,
                               "max_depth": args.max_depth, "min_samples_leaf": args.min_samples_leaf,
                               "random_state": args.random_state}}}
    (Path(args.outdir) / "meta.json").write_text(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
