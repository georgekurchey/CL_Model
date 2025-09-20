import os, csv, json, math, argparse, time
from models.gbt_quantile import GBTQuantileStack
from models.arx_ewma import ARX_EWMA

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
    with open(path,"w",encoding="utf-8") as f:
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

def empirical_quantile(vals, p):
    b=sorted(vals); n=len(b)
    if n==0: return 0.0
    pos=p*(n-1)
    lo=int(math.floor(pos)); hi=int(math.ceil(pos))
    if lo==hi: return b[lo]
    w=pos-lo; return b[lo]*(1-w)+b[hi]*w

def enforce_monotone(qrow):
    out=list(qrow)
    for i in range(1,len(out)):
        if out[i] < out[i-1]:
            out[i] = out[i-1]
    return out

def cqr_sigma_bisect(q05_cal, ycal, sigma_cal, target=0.05):
    eps=1e-12
    s_norm=[(q05_cal[i]-ycal[i])/max(abs(sigma_cal[i]),eps) for i in range(len(ycal))]
    def cov(alpha):
        c=empirical_quantile(s_norm, 1.0-alpha)
        thr=[q05_cal[i]-c*sigma_cal[i] for i in range(len(ycal))]
        cnt=sum(1 for i in range(len(ycal)) if ycal[i] <= thr[i])
        return cnt/float(max(1,len(ycal))), c
    lo,hi=0.005,0.20
    cov_lo,_=cov(lo); cov_hi,_=cov(hi)
    for _ in range(30):
        mid=(lo+hi)/2.0
        cmid,_c=cov(mid)
        if cmid>target: lo=mid
        else: hi=mid
    alpha=(lo+hi)/2.0
    cov_mid,c=cov(alpha)
    return c, alpha, cov_mid

def run(config_path, folds):
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
        q_tr = gbt.predict(Xtr)
        q_te = gbt.predict(Xte)
        ntr=len(Xtr); ncal=max(50, int(0.40*ntr))
        Xcal = Xtr[ntr-ncal:]
        ycal = ytr[ntr-ncal:]
        q_cal = q_tr[ntr-ncal:]
        _, sigma_cal = m1.predict_params(Xcal)
        q05_cal = [row[idx05] for row in q_cal]
        c_sigma, alpha_star, cov_cal = cqr_sigma_bisect(q05_cal, ycal, sigma_cal, target=0.05)
        q05_te  = [row[idx05] for row in q_te]
        q50_te  = [row[idx50] for row in q_te]
        q05_cqr = [min(q05_te[i]-c_sigma*sigma_te[i], q50_te[i]) for i in range(len(q05_te))]
        header=["date","model","y","mu","sigma"]+[f"q{int(t*100):02d}" for t in taus]
        z=[inv_norm_cdf(t) for t in taus]
        out=[]
        for i in range(len(yte)):
            qn=[mu_te[i]+sigma_te[i]*zz for zz in z]
            out.append([dte[i],"arx_ewma", f"{yte[i]:.8f}", f"{mu_te[i]:.8f}", f"{sigma_te[i]:.8f}"]+[f"{v:.8f}" for v in qn])
            base_row = [max(min(v,0.2), -0.2) for v in q_te[i]]
            base_row = enforce_monotone(base_row)
            out.append([dte[i],"gbt_quantile", f"{yte[i]:.8f}","",""]+[f"{v:.8f}" for v in base_row])
            cqr_row = list(base_row)
            cqr_row[idx05] = max(min(q05_cqr[i],0.2), -0.2)
            cqr_row = enforce_monotone(cqr_row)
            out.append([dte[i],"gbt_cqr", f"{yte[i]:.8f}","",""]+[f"{v:.8f}" for v in cqr_row])
        out_path=os.path.join(cfg["paths"]["preds"], f"preds_fold{k}.csv")
        write_csv(out_path, header, out)
        print(f"[Fold {k}] wrote {out_path} | alpha*={alpha_star:.4f} cal_coverage={cov_cal:.4f} | time={time.time()-t0:.1f}s")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", default="/Users/georgekurchey/CL_Model/config/pipeline.json")
    ap.add_argument("--folds", type=int, default=5)
    a=ap.parse_args()
    run(a.config, a.folds)
