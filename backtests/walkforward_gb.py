
import argparse, json, os, csv, math
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

# ---------- main ----------
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--folds", type=int, default=5)
    a=ap.parse_args()

    cfg=json.load(open(a.config,"r",encoding="utf-8"))
    rows=read_features(os.path.join(cfg["paths"]["proc"],"features.csv"))
    cols, dates, X, y = make_Xy(rows)
    splits = date_splits(len(dates), cfg["walkforward"]["train_years"], cfg["walkforward"]["test_months"], a.folds)
    taus=[i/100.0 for i in range(5,100,5)]
    os.makedirs(cfg["paths"]["preds"], exist_ok=True)

    for k,(tr,te) in enumerate(splits, start=1):
        tr0,tr1=tr; te0,te1=te
        Xtr,ytr = X[tr0:tr1], y[tr0:tr1]
        Xte,yte = X[te0:te1], y[te0:te1]
        dte = dates[te0:te1]

        # Baseline Normal: ARX + EWMA
        m1 = ARX_EWMA(lmbda=0.94, ridge=1e-2).fit(Xtr, ytr)
        mu, sigma = m1.predict_params(Xte)

        # ==== Gradient Boosting Quantiles ====
        # Train/calibration split (last 30% of train for calibration)
        ntr=len(Xtr); ncal=max(50, int(0.3*ntr))
        if ncal>=ntr: ncal=max(1,int(0.2*ntr))
        nfit=max(1, ntr-ncal)
        Xfit, yfit = Xtr[:nfit], ytr[:nfit]
        Xcal, ycal = Xtr[nfit:], ytr[nfit:]

        gbt = GBTQuantileStack(taus, n_estimators=600, learning_rate=0.05, max_depth=3, min_samples_leaf=20, random_state=42)
        gbt.fit(Xfit, yfit)

        q_cal = gbt.predict(Xcal)
        q_te  = gbt.predict(Xte)

        # Probability-level calibration (tau mapping) on calibration slice
        r = coverage_curve(ycal, q_cal)
        r = monotone_nondec(r)
        tau_map = [invert_map(taus, r, t) for t in taus]

        # Apply tau-map to test rows via interpolation within the boosted quantile fan
        q_te_tau = []
        for row in q_te:
            # enforce monotone first (safety)
            row = sorted(row)
            qrow=[]
            for tstar in tau_map:
                qrow.append(interp_q(row, taus, tstar))
            # final safety: clip daily returns; enforce monotone
            qrow = [max(min(v,0.2), -0.2) for v in qrow]
            q_te_tau.append(sorted(qrow))

        # ==== Write predictions ====
        z=[inv_norm_cdf(t) for t in taus]
        header=["date","model","y","mu","sigma"]+[f"q{int(t*100):02d}" for t in taus]
        out=[]
        for i in range(len(yte)):
            qn=[mu[i]+sigma[i]*zz for zz in z]
            out.append([dte[i],"arx_ewma", f"{yte[i]:.8f}", f"{mu[i]:.8f}", f"{sigma[i]:.8f}"]+[f"{v:.8f}" for v in qn])
            out.append([dte[i],"gbt_quantile", f"{yte[i]:.8f}","",""]+[f"{v:.8f}" for v in q_te_tau[i]])

        write_csv(os.path.join(cfg["paths"]["preds"], f"preds_fold{k}.csv"), header, out)
        print("Wrote", os.path.join(cfg["paths"]["preds"], f"preds_fold{k}.csv"))
