import argparse, json, os, csv, math
from models.arx_ewma import ARX_EWMA
from models.quantile_reg import QuantileStack

# ---------- IO ----------
def read_features(path):
    with open(path,"r",encoding="utf-8") as f:
        r=csv.DictReader(f); rows=list(r)
    rows.sort(key=lambda x: x["date"])
    return rows

def make_Xy(rows):
    cols=[f"cl_settle_m{k}" for k in range(1,13)] + [
        "usd_broad_index","ust_10y","breakeven_10y","wti_cvol_1m",
        "eia_day_dummy","inv_surprise","spread_m1_m2"
    ]
    X=[]; y=[]; dates=[]; prev=None
    for row in rows:
        xi=[]
        for c in cols:
            if prev is not None:
                try: v=float(prev[c])
                except: v=0.0
            else:
                try: v=float(row[c])
                except: v=0.0
            xi.append(v)
        X.append(xi)
        try: y.append(float(row["ret_1d"]))
        except: y.append(0.0)
        dates.append(row["date"]); prev=row
    return cols, dates, X, y

def date_splits(n, train_years=3, test_months=6, max_folds=5):
    train_n=int(252*train_years); test_n=int(21*test_months)
    splits=[]; start_train=0
    while len(splits)<max_folds:
        start_test=start_train+train_n; end_test=start_test+test_n
        if end_test>n-5: break
        splits.append(((start_train,start_test),(start_test,end_test)))
        start_train += test_n
    return splits

def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header)+"\n")
        for r in rows:
            f.write(",".join(r)+"\n")

# ---------- helpers ----------
def inv_norm_cdf(p):
    a1=-39.69683028665376; a2=220.9460984245205; a3=-275.9285104469687
    a4=138.3577518672690; a5=-30.66479806614716; a6=2.506628277459239
    b1=-54.47609879822406; b2=161.5858368580409; b3=-155.6989798598866
    b4=66.80131188771972; b5=-13.28068155288572
    c1=-0.007784894002430293; c2=-0.3223964580411365
    c3=-2.400758277161838; c4=-2.549732539343734
    c5=4.374664141464968; c6=2.938163982698783
    d1=0.007784695709041462; d2=0.3224671290700398
    d3=2.445134137142996; d4=3.754408661907416
    plow=0.02425; phigh=1-plow
    if p<plow:
        q=(-2*math.log(p))**0.5
        return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6)/((((d1*q+d2)*q+d3)*q+d4)*q+1)
    if p>phigh:
        q=(-2*math.log(1-p))**0.5
        return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6)/((((d1*q+d2)*q+d3)*q+d4)*q+1)
    q=p-0.5; r=q*q
    return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q/(((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1)

def _mean(vals): return sum(vals)/len(vals) if vals else 0.0
def _sdev(vals, m):
    if not vals: return 1.0
    var = sum((v-m)*(v-m) for v in vals) / max(1, len(vals)-1)
    return var**0.5 if var>0 else 1.0

def zfit(X, y):
    cols = list(zip(*X)) if X else []
    muX = [_mean(col) for col in cols] if cols else []
    sdX = [_sdev(col, m) for col, m in zip(cols, muX)] if cols else []
    my  = _mean(y); sy  = _sdev(y, my)
    sdX = [s if s>0 else 1.0 for s in sdX]; sy = sy if sy>0 else 1.0
    return muX, sdX, my, sy

def zapply_X(X, mu, sd):
    Z=[]
    for row in X:
        Z.append([(v-m)/s for v, m, s in zip(row, mu, sd)])
    return Z

def zapply_y(y, my, sy): return [(v-my)/sy for v in y]

def empirical_quantile(a, q):
    if not a: return 0.0
    b = sorted(a); pos = q*(len(b)-1)
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo==hi: return b[lo]
    w = pos-lo; return b[lo]*(1-w)+b[hi]*w

# ---------- main ----------
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--folds", type=int, default=5)
    a=ap.parse_args()

    cfg=json.load(open(a.config,"r",encoding="utf-8"))
    proc = cfg["paths"]["proc"]
    rows=read_features(os.path.join(proc,"features.csv"))
    cols, dates, X, y = make_Xy(rows)

    splits = date_splits(len(dates), cfg["walkforward"]["train_years"], cfg["walkforward"]["test_months"], a.folds)
    taus=[i/100.0 for i in range(5,100,5)]
    idx05 = taus.index(0.05); idx95 = taus.index(0.95)
    os.makedirs(cfg["paths"]["preds"], exist_ok=True)

    TAIL_ALPHA = 0.05
    ETA = 0.12           # learning rate for online offsets
    E_MAX = 3.0          # cap in z-space (std-residual space)
    PHI_MIN, PHI_MAX = 0.5, 3.0

    for k,(tr,te) in enumerate(splits, start=1):
        tr0,tr1=tr; te0,te1=te
        Xtr,ytr = X[tr0:tr1], y[tr0:tr1]
        Xte,yte = X[te0:te1], y[te0:te1]
        dte = dates[te0:te1]

        # Baseline Normal params (for report + sigma for std.)
        m1 = ARX_EWMA(lmbda=0.94, ridge=1e-2).fit(Xtr, ytr)
        mu_tr, sigma_tr = m1.predict_params(Xtr)
        mu_te, sigma_te = m1.predict_params(Xte)

        # Standardize target by sigma (heteroskedasticity removal)
        eps = 1e-8
        ytr_std = [ytr[i]/max(sigma_tr[i],eps) for i in range(len(ytr))]

        # Standardize features + y_std to zero-mean unit-variance (z-space)
        muX, sdX, my, sy = zfit(Xtr, ytr_std)
        Xtr_z = zapply_X(Xtr, muX, sdX)
        ytr_z = zapply_y(ytr_std, my, sy)
        Xte_z  = zapply_X(Xte,  muX, sdX)

        # Split train into fit/cal
        ntr=len(Xtr_z)
        ncal=max(50, int(0.3*ntr))
        if ncal>=ntr: ncal=max(1,int(0.2*ntr))
        nfit=max(1, ntr-ncal)
        Xfit, yfit = Xtr_z[:nfit], ytr_z[:nfit]
        Xcal, ycal = Xtr_z[nfit:], ytr_z[nfit:]

        # Fit quantiles on standardized z-space
        q_model = QuantileStack(taus, lr=0.01, epochs=25, l2=1e-2, batch=128).fit(Xfit, yfit)

        # Per-quantile additive calibration on cal slice (z-space)
        q_cal = q_model.predict(Xcal)
        deltas=[]
        for j,tau in enumerate(taus):
            res=[ycal[i]-q_cal[i][j] for i in range(len(ycal))]
            deltas.append(empirical_quantile(res, tau))

        # Predict test base fan (z-space) and add deltas
        q_te = q_model.predict(Xte_z)
        q_te_adj = [[q_te[i][j]+deltas[j] for j in range(len(taus))] for i in range(len(q_te))]

        # Volatility scaling factors relative to cal-average sigma
        base = _mean(sigma_tr[nfit:]) if nfit < len(sigma_tr) else _mean(sigma_tr)
        base = max(base, 1e-8)
        phi_te=[max(PHI_MIN, min(s/base, PHI_MAX)) for s in sigma_te]

        # Prepare sequential outputs with ONLINE per-tail offsets (z-space)
        out=[]; header=["date","model","y","mu","sigma"]+[f"q{int(t*100):02d}" for t in taus]
        z=[inv_norm_cdf(t) for t in taus]

        # Precompute yte in z-space (to update offsets using only past)
        # y_std = y / sigma_te; y_z = (y_std - my)/sy
        yte_std=[yte[i]/max(sigma_te[i],eps) for i in range(len(yte))]
        yte_z=[(v - my)/sy for v in yte_std]

        # Initialize offsets
        e_lo = 0.0
        e_hi = 0.0

        for i in range(len(yte)):
            # Model 1 (Normal) for this date
            qn=[mu_te[i]+sigma_te[i]*zz for zz in z]
            out.append([dte[i],"arx_ewma", f"{yte[i]:.8f}", f"{mu_te[i]:.8f}", f"{sigma_te[i]:.8f}"]+[f"{v:.8f}" for v in qn])

            # Base fan at i (z-space) and apply current ONLINE offsets to tails only
            row = q_te_adj[i][:]  # copy
            row[idx05] -= (e_lo * phi_te[i])
            row[idx95] += (e_hi * phi_te[i])

            # Safety: monotone after tail tweak
            row = sorted(row)

            # Un-standardize to raw return space: undo z, then multiply by sigma_te[i]
            row_u = [v*sy + my for v in row]
            r = max(sigma_te[i], eps)
            row_raw = [max(min(v*r, 0.2), -0.2) for v in row_u]  # clip extreme daily returns
            row_raw = sorted(row_raw)

            # Write quantile row
            out.append([dte[i],"quantile_regression", f"{yte[i]:.8f}","",""]+[f"{v:.8f}" for v in row_raw])

            # ---- ONLINE UPDATE (uses only past + today) ----
            # Compare in SAME z-space we updated in:
            # We need today's produced q05/q95 in z-space (before unscale).
            # Recompute those quickly:
            q05_z = q_te_adj[i][idx05] - (e_lo * phi_te[i])
            q95_z = q_te_adj[i][idx95] + (e_hi * phi_te[i])

            miss_lo = 1.0 if yte_z[i] < q05_z else 0.0
            miss_hi = 1.0 if yte_z[i] > q95_z else 0.0

            # Multiplicative weights update keeps positivity and adapts fast
            e_lo = min(E_MAX, max(0.0, e_lo * math.exp(ETA*(miss_lo - TAIL_ALPHA))))
            e_hi = min(E_MAX, max(0.0, e_hi * math.exp(ETA*(miss_hi - TAIL_ALPHA))))

        # Save fold predictions
        write_csv(os.path.join(cfg["paths"]["preds"], f"preds_fold{k}.csv"), header, out)
        print("Wrote", os.path.join(cfg["paths"]["preds"], f"preds_fold{k}.csv"))
