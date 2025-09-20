
import os, csv, json, math, argparse, time
from collections import defaultdict

ROOT="/Users/georgekurchey/CL_Model"

# ----------------- common utils -----------------
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

def norm_cdf(z):
    import math
    return 0.5*(1.0 + math.erf(z/math.sqrt(2.0)))

def empirical_quantile(vals, p):
    if not vals: return 0.0
    b=sorted(vals); n=len(b); pos=p*(n-1)
    lo=int(math.floor(pos)); hi=int(math.ceil(pos))
    if lo==hi: return b[lo]
    w=pos-lo; return b[lo]*(1-w)+b[hi]*w

def enforce_monotone(qrow):
    out=list(qrow)
    for i in range(1,len(out)):
        if out[i] < out[i-1]:
            out[i] = out[i-1]
    return out

from models.gbt_quantile import GBTQuantileStack
from models.arx_ewma import ARX_EWMA
from sklearn.tree import DecisionTreeClassifier
from collections import defaultdict

def learn_treewalk_deltas(Xcal, ycal, q05_cal, max_depth=3, min_leaf=50, target=0.05, tol=0.002, cap=0.05):
    z = [1 if ycal[i] <= q05_cal[i] else 0 for i in range(len(ycal))]
    if len(set(z))<=1:
        return {"tree": None, "deltas": {}, "scale": 0.0}
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_leaf, random_state=42)
    clf.fit(Xcal, z)
    leaf_ids = clf.apply(Xcal)
    by_leaf = defaultdict(list); by_leaf_flags = defaultdict(list)
    res = [ycal[i]-q05_cal[i] for i in range(len(ycal))]
    for i,lid in enumerate(leaf_ids):
        by_leaf[lid].append(res[i])
        by_leaf_flags[lid].append(z[i])
    deltas = {}
    for lid, arr in by_leaf.items():
        p_leaf = sum(by_leaf_flags[lid]) / float(len(by_leaf_flags[lid]))
        if p_leaf > target + tol:
            d = empirical_quantile(arr, target)
            d = max(-cap, min(cap, d))
            deltas[lid] = min(0.0, d)
        else:
            deltas[lid] = 0.0
    def coverage_with_scale(s):
        cnt=0; n=len(ycal)
        for i,lid in enumerate(leaf_ids):
            q = q05_cal[i] + s*deltas[lid]
            if ycal[i] <= q: cnt+=1
        return cnt/float(max(1,n))
    lo,hi=0.0,1.0
    for _ in range(25):
        mid=(lo+hi)/2.0
        cov = coverage_with_scale(mid)
        if cov > target: lo=mid
        else: hi=mid
    s = (lo+hi)/2.0
    return {"tree": clf, "deltas": deltas, "scale": s}

def run(config_path, folds, max_depth, min_leaf, cap, eta):
    cfg=json.load(open(config_path,"r",encoding="utf-8"))
    rows=read_features(os.path.join(cfg["paths"]["proc"],"features.csv"))
    cols, dates, X, y = make_Xy(rows)
    splits = date_splits(len(dates), cfg["walkforward"]["train_years"], cfg["walkforward"]["test_months"], folds)
    taus=[i/100.0 for i in range(5,100,5)]
    idx05=taus.index(0.05); idx50=taus.index(0.50)
    os.makedirs(cfg["paths"]["preds"], exist_ok=True)
    for k,(tr,te) in enumerate(splits, start=1):
        t0=time.time()
        tr0,tr1=tr; te0,te1=te
        Xtr,ytr = X[tr0:tr1], y[tr0:tr1]
        Xte,yte = X[te0:te1], y[te0:te1]
        dte = dates[te0:te1]
        gbt = GBTQuantileStack(taus, n_estimators=800, learning_rate=0.05,
                               max_depth=3, min_samples_leaf=20,
                               random_state=42, early_stopping=True,
                               n_iter_no_change=20, validation_fraction=0.1)
        gbt.fit(Xtr, ytr)
        m1 = ARX_EWMA(lmbda=0.94, ridge=1e-2).fit(Xtr, ytr)
        mu_te, sigma_te = m1.predict_params(Xte)
        mu_cal, sigma_cal = m1.predict_params(Xtr[int(len(Xtr)*0.70):])
        ntr=len(Xtr); ncal=max(50, int(0.30*ntr))
        Xcal, ycal = Xtr[ntr-ncal:], ytr[ntr-ncal:]
        _, sigma_cal = m1.predict_params(Xcal)
        _, sigma_cal = m1.predict_params(Xcal)
        q_tr = gbt.predict(Xtr)
        q_cal = q_tr[ntr-ncal:]
        q_te  = gbt.predict(Xte)
        q05_cal = [row[idx05] for row in q_cal]
        q50_cal = [row[idx50] for row in q_cal]
        q05_te  = [row[idx05] for row in q_te]
        q50_te  = [row[idx50] for row in q_te]
        tw = learn_treewalk_deltas(Xcal, ycal, q05_cal, max_depth=max_depth, min_leaf=min_leaf, cap=cap)
        clf = tw.get("tree"); deltas = tw.get("deltas", {}); scale = float(tw.get("scale", 0.0))
        # ---- global sigma-based calibration c_sigma ----
        if clf is not None:
            leaf_cal = clf.apply(Xcal)
        else:
            leaf_cal = [0]*len(Xcal)
        def _cov_sigma(c):
            n=len(ycal); cnt=0
            for i in range(n):
                delta = deltas.get(leaf_cal[i],0.0) * scale
                q = q05_cal[i] + delta + c * sigma_cal[i]
                if ycal[i] <= q: cnt+=1
            return cnt/float(max(1,n))
        lo,hi=-1.5,1.5
        for _ in range(30):
            mid=(lo+hi)/2.0
            cov=_cov_sigma(mid)
            if cov>0.05: hi=mid
            else: lo=mid
        c_sigma=(lo+hi)/2.0
        if clf is None:
            leaf_te = [0]*len(Xte)
        else:
            leaf_te = clf.apply(Xte)
        med_cal = sorted(sigma_cal)[len(sigma_cal)//2] if sigma_cal else 1.0
        # precompute per-sample leaf delta on TEST (vol-aware)
        d_leaf_te=[]
        for i in range(len(Xte)):
            lid = leaf_te[i] if clf is not None else 0
            d = deltas.get(lid,0.0) * scale
            if med_cal>1e-12:
                vr = sigma_te[i]/med_cal
                vr = max(0.5, min(2.0, vr))
                d = d * ((1.0-eta) + eta*vr)
            d_leaf_te.append(d)

        # expected-coverage match on TEST FEATURES (no leakage): choose c_pred so mean Phi(...) = 0.05
        def mean_prob(c):
            n=len(Xte)
            if n==0: return 0.0
            ssum=0.0
            for i in range(n):
                thr = q05_te[i] + d_leaf_te[i] + (c_sigma + c)*sigma_te[i]
                z = (thr - mu_te[i]) / (sigma_te[i] if abs(sigma_te[i])>1e-12 else 1.0)
                ssum += norm_cdf(z)
            return ssum/float(n)

        lo,hi=-2.0,2.0
        for _ in range(32):
            mid=(lo+hi)/2.0
            p=mean_prob(mid)
            if p>0.05: hi=mid
            else: lo=mid
        c_pred=(lo+hi)/2.0

        out_corr=[]
        for i in range(len(Xte)):
            lid = leaf_te[i] if clf is not None else 0
            base = q05_te[i]
            d_leaf = deltas.get(lid,0.0) * scale
            if med_cal>1e-12:
                vr = sigma_te[i]/med_cal
                vr = max(0.5, min(2.0, vr))
                d_leaf = d_leaf * ((1.0-eta) + eta*vr)
            q05_adj = base + d_leaf + c_sigma * sigma_te[i]
            q05_adj = min(q05_adj, q50_te[i])
            out_corr.append(q05_adj)
        rows_out=[]
        header=["date","model","y","mu","sigma"]+[f"q{int(t*100):02d}" for t in taus]
        z=[inv_norm_cdf(t) for t in taus]
        for i in range(len(yte)):
            qn=[mu_te[i]+sigma_te[i]*zz for zz in z]
            rows_out.append([dte[i],"arx_ewma", f"{yte[i]:.8f}", f"{mu_te[i]:.8f}", f"{sigma_te[i]:.8f}"]+[f"{v:.8f}" for v in qn])
            base_row = [max(min(v,0.2), -0.2) for v in q_te[i]]
            base_row = enforce_monotone(base_row)
            rows_out.append([dte[i],"gbt_quantile", f"{yte[i]:.8f}","",""]+[f"{v:.8f}" for v in base_row])
            tw_row = list(base_row)
            tw_row[idx05] = max(min(out_corr[i],0.2), -0.2)
            tw_row = enforce_monotone(tw_row)
            rows_out.append([dte[i],"gbt_treewalk", f"{yte[i]:.8f}","",""]+[f"{v:.8f}" for v in tw_row])
        out_path=os.path.join("/Users/georgekurchey/CL_Model/preds", f"preds_fold{k}.csv")
        write_csv(out_path, header, rows_out)
        print(f"[Fold {k}] wrote {out_path} | scale={scale:.3f}, c_sigma={c_sigma:.3f}, c_pred={c_pred:.3f} | time={time.time()-t0:.1f}s")

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max_depth", type=int, default=3)
    ap.add_argument("--min_leaf", type=int, default=50)
    ap.add_argument("--cap", type=float, default=0.05)
    ap.add_argument("--eta", type=float, default=0.5)
    a=ap.parse_args()
    run(a.config, a.folds, a.max_depth, a.min_leaf, a.cap, a.eta)
