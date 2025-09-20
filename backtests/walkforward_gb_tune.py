
import argparse, json, os, csv, math, itertools, time
from models.arx_ewma import ARX_EWMA
from models.gbt_quantile import GBTQuantileStack

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

# ---------- math ----------
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

def coverage_curve(y_vals, q_grid):
    if not q_grid: return []
    m=len(q_grid[0]); n=len(q_grid)
    r=[0.0]*m
    for j in range(m):
        c=0
        for i in range(n):
            if y_vals[i] <= q_grid[i][j]: c+=1
        r[j] = c/float(n) if n>0 else 0.0
    return r

def monotone_nondec(a):
    out=[]; cur=-1e9
    for v in a:
        if v<cur: out.append(cur)
        else: out.append(v); cur=v
    return [0.0 if v<0 else 1.0 if v>1 else v for v in out]

def invert_map(tau_grid, r_grid, target_tau):
    eps=1e-12
    T=[0.0]+tau_grid+[1.0]; R=[0.0]+r_grid+[1.0]
    for k in range(len(R)-1):
        if R[k] <= target_tau <= R[k+1]:
            if R[k+1]-R[k] < eps: return T[k]
            w=(target_tau - R[k])/(R[k+1]-R[k])
            return T[k] + w*(T[k+1]-T[k])
    return T[-1]

def interp_q(row, tau_grid, t):
    # row: quantiles at tau_grid indices
    if len(tau_grid)==1: return row[0]
    if t <= tau_grid[0]:
        t0,t1=tau_grid[0],tau_grid[1]; q0,q1=row[0],row[1]
        return q0 + (q1-q0)*(t-t0)/(t1-t0)
    if t >= tau_grid[-1]:
        t0,t1=tau_grid[-2],tau_grid[-1]; q0,q1=row[-2],row[-1]
        return q1 + (q1-q0)*(t-t1)/(t1-t0)
    lo,hi=0,len(tau_grid)-1
    while hi-lo>1:
        mid=(lo+hi)//2
        if tau_grid[mid] <= t: lo=mid
        else: hi=mid
    t0,t1=tau_grid[lo],tau_grid[hi]
    q0,q1=row[lo],row[hi]
    w=(t-t0)/(t1-t0)
    return q0 + w*(q1-q0)

def pinball_loss(y, q, tau):
    s=0.0; n=len(y)
    for yi,qi in zip(y,q):
        e=yi-qi
        s += (tau*e) if e>=0 else ((tau-1.0)*e)
    return s/float(max(1,n))

def empirical_quantile(vals, p):
    if not vals: return 0.0
    b=sorted(vals); n=len(b); pos=p*(n-1)
    lo=int(math.floor(pos)); hi=int(math.ceil(pos))
    if lo==hi: return b[lo]
    w=pos-lo; return b[lo]*(1-w)+b[hi]*w

def enforce_monotone(qrow):
    # make non-decreasing without reindexing
    out = list(qrow)
    for i in range(1, len(out)):
        if out[i] < out[i-1]:
            out[i] = out[i-1]
    return out

def split_fit_map_val(X, y):
    n=len(X)
    n_fit=max(1, int(round(0.60*n)))
    n_map=max(25, int(round(0.20*n)))
    if n_fit+n_map>=n: n_map=max(10, int(0.15*n))
    n_val=max(10, n - (n_fit+n_map))
    if n_fit+n_map+n_val>n:
        n_val = n - (n_fit+n_map)
    Xfit, yfit = X[:n_fit], y[:n_fit]
    Xmap, ymap = X[n_fit:n_fit+n_map], y[n_fit:n_fit+n_map]
    Xval, yval = X[n_fit+n_map:], y[n_fit+n_map:]
    return (Xfit,yfit),(Xmap,ymap),(Xval,yval)

def score_combo(q_map, y_map, q_val, y_val, base_taus, target_taus, w_cov=14.0):
    r = coverage_curve(y_map, q_map)
    r = monotone_nondec(r)
    tau_map = [invert_map(base_taus, r, t) for t in target_taus]

    q_val_m=[]
    for row in q_val:
        row = enforce_monotone(row)
        q_val_m.append([interp_q(row, base_taus, tstar) for tstar in tau_map])

    idx05 = target_taus.index(0.05); idx95 = target_taus.index(0.95)
    n=len(y_val)
    cov05 = sum(1 for i in range(n) if y_val[i] <= q_val_m[i][idx05]) / float(max(1,n))
    cov95 = sum(1 for i in range(n) if y_val[i] <= q_val_m[i][idx95]) / float(max(1,n))
    cov_err = abs(cov05-0.05) + abs(0.95-cov95)

    q05=[row[idx05] for row in q_val_m]
    q95=[row[idx95] for row in q_val_m]
    pl = 0.5*(pinball_loss(y_val,q05,0.05) + pinball_loss(y_val,q95,0.95))

    return w_cov*cov_err + pl, {"cov05":cov05,"cov95":cov95,"pinball":pl}

# ---- multiplicative tail scaling (relative to median) ----
def find_k_low(y, q_low, q_med, target, kmin=0.2, kmax=5.0, iters=25):
    # For lower tail: coverage decreases as k increases.
    lo, hi = kmin, kmax
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        thr = [qm + mid*(ql - qm) for ql, qm in zip(q_low, q_med)]
        cov = sum(1 for yi,ti in zip(y,thr) if yi <= ti) / float(max(1,len(y)))
        if cov > target:
            lo = mid  # need lower coverage -> increase k
        else:
            hi = mid  # need higher coverage -> decrease k
    return max(kmin, min(kmax, (lo+hi)/2.0))

def find_k_high(y, q_high, q_med, target, kmin=0.2, kmax=5.0, iters=25):
    # For upper tail: coverage increases as k increases.
    lo, hi = kmin, kmax
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        thr = [qm + mid*(qh - qm) for qh, qm in zip(q_high, q_med)]
        cov = sum(1 for yi,ti in zip(y,thr) if yi <= ti) / float(max(1,len(y)))
        if cov < target:
            lo = mid  # need higher coverage -> increase k
        else:
            hi = mid  # need lower coverage -> decrease k
    return max(kmin, min(kmax, (lo+hi)/2.0))

# ---------- main ----------
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fast", action="store_true", help="Use a smaller hyperparameter grid")
    ap.add_argument("--heavy", action="store_true", help="Use a heavier grid (slower, better coverage)")
    a=ap.parse_args()

    cfg=json.load(open(a.config,"r",encoding="utf-8"))
    rows=read_features(os.path.join(cfg["paths"]["proc"],"features.csv"))
    cols, dates, X, y = make_Xy(rows)
    splits = date_splits(len(dates), cfg["walkforward"]["train_years"], cfg["walkforward"]["test_months"], a.folds)
    taus=[i/100.0 for i in range(5,100,5)]
    SEARCH_TAUS=[0.05,0.95]
    os.makedirs(cfg["paths"]["preds"], exist_ok=True)

    # Hyperparameter grid
    if a.heavy:
        grid = {"n_estimators":[800,1200,1800],"learning_rate":[0.03,0.05],
                "max_depth":[3,4,5],"min_samples_leaf":[10,20,40]}
    elif a.fast:
        grid = {"n_estimators":[600,1000],"learning_rate":[0.03,0.05],
                "max_depth":[3,4],"min_samples_leaf":[20,40]}
    else:
        grid = {"n_estimators":[400,800],"learning_rate":[0.03,0.05],
                "max_depth":[3,4],"min_samples_leaf":[20,40]}
    combos=list(itertools.product(grid["n_estimators"], grid["learning_rate"],
                                  grid["max_depth"], grid["min_samples_leaf"]))

    for k,(tr,te) in enumerate(splits, start=1):
        t0=time.time()
        tr0,tr1=tr; te0,te1=te
        Xtr,ytr = X[tr0:tr1], y[tr0:tr1]
        Xte,yte = X[te0:te1], y[te0:te1]
        dte = dates[te0:te1]

        # Baseline Normal: ARX + EWMA
        m1 = ARX_EWMA(lmbda=0.94, ridge=1e-2).fit(Xtr, ytr)
        mu, sigma = m1.predict_params(Xte)

        # Split train -> fit/map/val
        (Xfit,yfit),(Xmap,ymap),(Xval,yval) = split_fit_map_val(Xtr, ytr)

        # ---- Hyperparameter search (tails only) ----
        best=None; best_score=1e9; best_metrics=None
        for (ne,lr,md,ml) in combos:
            mdl = GBTQuantileStack(SEARCH_TAUS, n_estimators=ne, learning_rate=lr,
                                   max_depth=md, min_samples_leaf=ml, random_state=42,
                                   early_stopping=True, n_iter_no_change=20, validation_fraction=0.1)
            mdl.fit(Xfit, yfit)
            q_map = mdl.predict(Xmap)
            q_val = mdl.predict(Xval)
            s, met = score_combo(q_map, ymap, q_val, yval, SEARCH_TAUS, taus, w_cov=14.0)
            if s < best_score:
                best_score=s; best=(ne,lr,md,ml); best_metrics=met

        print(f"[Fold {k}] best params: ne={best[0]}, lr={best[1]}, depth={best[2]}, leaf={best[3]} | "
              f"val cov05={best_metrics['cov05']:.3f}, cov95={best_metrics['cov95']:.3f}, pinball={best_metrics['pinball']:.6f}")

        # ---- Retrain on FULL train with best params, full quantile grid ----
        ne,lr,md,ml = best
        gbt = GBTQuantileStack(taus, n_estimators=ne, learning_rate=lr,
                               max_depth=md, min_samples_leaf=ml, random_state=42,
                               early_stopping=True, n_iter_no_change=20, validation_fraction=0.1)
        gbt.fit(Xtr, ytr)

        # Build additive residual calibration on last 30% of train
        ntr=len(Xtr); ncal=max(50, int(0.30*ntr))
        if ncal>=ntr: ncal=max(1,int(0.20*ntr))
        Xcal, ycal = Xtr[ntr-ncal:], ytr[ntr-ncal:]
        q_cal = gbt.predict(Xcal)

        # per-quantile residual shift: delta_j = Quantile_tau(y - q_cal_j)
        deltas=[]
        for j,tau in enumerate(taus):
            res=[ycal[i]-q_cal[i][j] for i in range(len(ycal))]
            deltas.append(empirical_quantile(res, tau))
        q_cal_add = [[q_cal[i][j]+deltas[j] for j in range(len(taus))] for i in range(len(q_cal))]

        # coverage-driven tau-map on corrected cal
        r = coverage_curve(ycal, q_cal_add)
        r = monotone_nondec(r)
        tau_map = [invert_map(taus, r, t) for t in taus]

        # Map cal and compute multiplicative scalers relative to median
        idx05=taus.index(0.05); idx50=taus.index(0.50); idx95=taus.index(0.95)
        q_cal_mapped=[]
        for row in q_cal_add:
            row = enforce_monotone(row)
            q_cal_mapped.append([interp_q(row, taus, tstar) for tstar in tau_map])
        q05c=[row[idx05] for row in q_cal_mapped]
        q50c=[row[idx50] for row in q_cal_mapped]
        q95c=[row[idx95] for row in q_cal_mapped]

        # current calibration coverage
        cov05_cal = sum(1 for yi,q in zip(ycal,q05c) if yi<=q)/float(max(1,len(ycal)))
        cov95_cal = sum(1 for yi,q in zip(ycal,q95c) if yi<=q)/float(max(1,len(ycal)))
        tol = 0.002
        if abs(cov05_cal - 0.05) <= tol:
            k_low = 1.0
        elif cov05_cal > 0.05:
            k_low = find_k_low(ycal, q05c, q50c, 0.05, kmin=1.0, kmax=5.0, iters=25)
        else:
            k_low = find_k_low(ycal, q05c, q50c, 0.05, kmin=0.2, kmax=1.0, iters=25)

        if abs(cov95_cal - 0.95) <= tol:
            k_high = 1.0
        elif cov95_cal < 0.95:
            k_high = find_k_high(ycal, q95c, q50c, 0.95, kmin=1.0, kmax=5.0, iters=25)
        else:
            k_high = find_k_high(ycal, q95c, q50c, 0.95, kmin=0.2, kmax=1.0, iters=25)

        # Volatility-aware adjustment using ARX_EWMA sigmas
        _, sigma_cal_arr = m1.predict_params(Xcal)
        med_cal = (sorted(sigma_cal_arr)[len(sigma_cal_arr)//2] if sigma_cal_arr else 1.0)
        med_test = (sorted(sigma)[len(sigma)//2] if sigma else 1.0)
        vol_ratio = med_test/med_cal if med_cal > 1e-12 else 1.0
        vol_ratio = max(0.5, min(2.0, vol_ratio))
        eta = 0.7
        adj = eta*vol_ratio + (1.0-eta)*1.0
        k_low  = 1.0 + (k_low  - 1.0) * adj
        k_high = 1.0 + (k_high - 1.0) * adj

        # Predict test, apply (additive deltas) -> tau-map -> multiplicative scaling
        q_te = gbt.predict(Xte)
        q_te_add = [[q_te[i][j]+deltas[j] for j in range(len(taus))] for i in range(len(q_te))]
        q_te_tau=[]
        for row in q_te_add:
            row = enforce_monotone(row)
            mapped = [interp_q(row, taus, tstar) for tstar in tau_map]
            q50 = mapped[idx50]
            mapped[idx05] = q50 + k_low  * (mapped[idx05] - q50)
            mapped[idx95] = q50 + k_high * (mapped[idx95] - q50)
            mapped=[max(min(v,0.2), -0.2) for v in mapped]
            mapped = enforce_monotone(mapped)
            q_te_tau.append(mapped)

        # ---- Write predictions ----
        z=[inv_norm_cdf(t) for t in taus]
        header=["date","model","y","mu","sigma"]+[f"q{int(t*100):02d}" for t in taus]
        out=[]
        for i in range(len(yte)):
            qn=[mu[i]+sigma[i]*zz for zz in z]
            out.append([dte[i],"arx_ewma", f"{yte[i]:.8f}", f"{mu[i]:.8f}", f"{sigma[i]:.8f}"]+[f"{v:.8f}" for v in qn])
            out.append([dte[i],"gbt_quantile", f"{yte[i]:.8f}","",""]+[f"{v:.8f}" for v in q_te_tau[i]])

        write_csv(os.path.join(cfg["paths"]["preds"], f"preds_fold{k}.csv"), header, out)
        print(f"[Fold {k}] wrote preds_fold{k}.csv in {time.time()-t0:.1f}s")
