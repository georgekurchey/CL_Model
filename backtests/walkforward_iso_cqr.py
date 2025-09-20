import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

ROOT = Path("/Users/georgekurchey/CL_Model")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.conformal import conformal_offsets, apply_tail_offsets

def parse_taus(s):
    if s:
        out=[]
        for tok in s.split(","):
            tok=tok.strip()
            if not tok: continue
            if tok.endswith("%"): out.append(float(tok[:-1])/100.0)
            else:
                x=float(tok); out.append(x/100.0 if x>1 else x)
        out=[t for t in out if 0.0<t<1.0]
        return sorted(set(out))
    return [0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95]

def crps_from_grid(y, q_grid, taus):
    taus=np.asarray(taus,float); y=np.asarray(y,float)
    if q_grid.size==0 or len(taus)==0: return np.zeros(len(y))
    w=np.empty_like(taus)
    if len(taus)==1: w[:]=1.0
    else:
        w[0]=(taus[1]-0)/2; w[-1]=(1-taus[-2])/2
        if len(taus)>2: w[1:-1]=(taus[2:]-taus[:-2])/2
    losses=[]
    for i,t in enumerate(taus):
        e=y-q_grid[:,i]
        losses.append(np.maximum(t*e,(t-1)*e))
    L=np.vstack(losses).T
    return (L*w).sum(axis=1)

def fit_qgbt(X,y,taus,n_estimators,max_depth,learning_rate,min_samples_leaf,seed):
    models={}
    for t in taus:
        est=GradientBoostingRegressor(loss="quantile",alpha=t,
                                      n_estimators=n_estimators,max_depth=max_depth,
                                      learning_rate=learning_rate,min_samples_leaf=min_samples_leaf,
                                      random_state=seed)
        est.fit(X,y); models[t]=est
    return models

def predict_grid(models,X,taus):
    preds=np.column_stack([models[t].predict(X) for t in taus]) if len(taus)>0 else np.zeros((len(X),0))
    if preds.size: preds=np.maximum.accumulate(preds,axis=1)
    return preds

def iso_curve(y_cal,q_cal,taus):
    cov=[float(np.mean(y_cal<=q_cal[:,i])) for i,_ in enumerate(taus)]
    iso=IsotonicRegression(y_min=0.0,y_max=1.0,increasing=True,out_of_bounds="clip")
    x=np.array(taus,float); y=np.array(cov,float)
    f=iso.fit(x,y); xs=f.predict(x); ys=x
    for i in range(1,len(xs)):
        if xs[i]<=xs[i-1]: xs[i]=xs[i-1]+1e-9
    return xs,ys

def apply_iso(q_pred,taus,xs_cov,ys_tau,targets):
    taus=np.array(taus,float); xs=np.array(xs_cov,float); ys=np.array(ys_tau,float)
    targets=np.array(targets,float)
    tau_eff=np.interp(targets,xs,ys)
    out=np.empty((q_pred.shape[0],len(targets)),float)
    for i in range(q_pred.shape[0]):
        out[i,:]=np.interp(tau_eff,taus,q_pred[i,:])
    return out

def main():
    ap=argparse.ArgumentParser(description="Walk-forward GBT + isotonic + optional split-conformal tails")
    ap.add_argument("--features",default=str(ROOT/"data/proc/features.csv"))
    ap.add_argument("--outdir",default=str(ROOT/"preds"))
    ap.add_argument("--report",default=str(ROOT/"reports/wf_cqr_report.html"))
    ap.add_argument("--folds",type=int,default=5)
    ap.add_argument("--val_days",type=int,default=126)
    ap.add_argument("--min_train_days",type=int,default=504)
    ap.add_argument("--taus",default="5,10,20,30,40,50,60,70,80,90,95")
    ap.add_argument("--n_estimators",type=int,default=600)
    ap.add_argument("--max_depth",type=int,default=3)
    ap.add_argument("--learning_rate",type=float,default=0.05)
    ap.add_argument("--min_samples_leaf",type=int,default=80)
    ap.add_argument("--calib_cfg",default=str(ROOT/"config/calibration.json"))
    args=ap.parse_args()

    # config gate
    use_conf=False; conf_targets=[0.05,0.95]; tail_only=True; iso_days=252
    cfgp=Path(args.calib_cfg)
    if cfgp.exists():
        import json
        try:
            cfg=json.loads(cfgp.read_text())
            use_conf=bool(cfg.get("use_conformal", use_conf))
            conf_targets=[float(x) for x in cfg.get("targets", conf_targets)]
            tail_only=bool(cfg.get("tail_only", tail_only))
            iso_days=int(cfg.get("iso_days", iso_days))
        except Exception:
            pass

    taus=parse_taus(args.taus)
    df=pd.read_csv(args.features,parse_dates=["date"]).sort_values("date")
    if "ret_1d" not in df.columns: raise SystemExit("features must include 'ret_1d' as target")
    X_all=df[[c for c in df.columns if c not in ("date","ret_1d")]].astype(float).fillna(0.0).values
    y_all=df["ret_1d"].astype(float).values
    dates=df["date"].values

    n=len(df); n_val_total=args.folds*args.val_days
    train_end0=n-n_val_total
    if train_end0<args.min_train_days:
        args.val_days=max(30,(n-args.min_train_days)//max(1,args.folds))
        n_val_total=args.folds*args.val_days; train_end0=n-n_val_total
        if train_end0<args.min_train_days: raise SystemExit("Not enough rows for folds/min_train_days/val_days")

    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    all_folds=[]; fold_rows=[]
    for k in range(args.folds):
        tr_end=train_end0+k*args.val_days; va_end=tr_end+args.val_days
        tr_idx=slice(0,tr_end); va_idx=slice(tr_end,va_end)
        ntr=tr_end; ncal=min(iso_days,ntr); cal_idx=slice(tr_end-ncal,tr_end)

        Xtr,ytr=X_all[tr_idx],y_all[tr_idx]
        Xcal,ycal=X_all[cal_idx],y_all[cal_idx]
        Xva,yva=X_all[va_idx],y_all[va_idx]
        d_va=dates[va_idx]

        models=fit_qgbt(Xtr,ytr,taus,args.n_estimators,args.max_depth,args.learning_rate,args.min_samples_leaf,seed=42+k)
        q_cal_raw=predict_grid(models,Xcal,taus)
        xs_cov,ys_tau=iso_curve(ycal,q_cal_raw,taus)

        q_va_raw=predict_grid(models,Xva,taus)
        q_va_iso=apply_iso(q_va_raw,taus,xs_cov,ys_tau,taus)

        if use_conf:
            q_cal_iso=apply_iso(q_cal_raw,taus,xs_cov,ys_tau,taus)
            offs=conformal_offsets(ycal,q_cal_iso,taus,conf_targets)
            q_va_adj=apply_tail_offsets(q_va_iso,taus,offs,tail_only=tail_only)
        else:
            q_va_adj=q_va_iso

        crps=crps_from_grid(yva,q_va_adj,taus)
        qnames=[f"q{int(t*100):02d}" for t in taus]
        df_fold=pd.DataFrame({"date":pd.to_datetime(d_va),"y":yva})
        for i,name in enumerate(qnames): df_fold[name]=q_va_adj[:,i]
        df_fold["crps"]=crps

        var5=int((df_fold["y"]<df_fold.get("q05",np.inf)).sum()) if "q05" in df_fold.columns else None
        var95=int((df_fold["y"]>df_fold.get("q95",-np.inf)).sum()) if "q95" in df_fold.columns else None

        df_fold.to_csv(outdir/f"preds_cqr_fold{k+1}.csv",index=False)
        all_folds.append(df_fold)
        fold_rows.append({"fold":k+1,"n_val":int(len(df_fold)),"crps_mean":float(np.mean(crps)),
                          "var5_exceed":var5,"var95_exceed":var95})

    agg=pd.concat(all_folds,ignore_index=True)
    agg.to_csv(outdir/"preds_walkforward_cqr.csv",index=False)

    ov={"n_total":int(len(agg)),
        "crps_mean":float(np.mean(agg["crps"])) if len(agg)>0 else float("nan"),
        "var5_exceed": int((agg["y"]<agg["q05"]).sum()) if "q05" in agg.columns else None,
        "var95_exceed": int((agg["y"]>agg["q95"]).sum()) if "q95" in agg.columns else None}

    html=[]
    html.append("<html><head><meta charset='utf-8'><title>WF — Iso + Split-Conformal</title>")
    html.append("<style>body{font-family:Arial,Helvetica,sans-serif;margin:20px;} table{border-collapse:collapse;} th,td{border:1px solid #ddd;padding:6px;} th{background:#f0f0f0;}</style></head><body>")
    html.append("<h2>CL_Model — Walk-Forward (Isotonic + Split-Conformal Tails)</h2>")
    html.append(f"<p>use_conformal={use_conf} targets={conf_targets} tail_only={tail_only}</p>")
    html.append("<h3>Per-fold</h3>"+pd.DataFrame(fold_rows).to_html(index=False))
    html.append("<h3>Overall</h3>"+pd.DataFrame([ov]).to_html(index=False))
    tail=agg.tail(15).copy(); tail["date"]=pd.to_datetime(tail["date"]).dt.strftime("%Y-%m-%d")
    html.append("<h3>Last 15 predictions</h3>"+tail.to_html(index=False))
    html.append("</body></html>")
    Path(args.report).write_text("\n".join(html),encoding="utf-8")
    print(f"[report] -> {args.report}")
    print(f"[wf] preds in {outdir}/preds_cqr_fold*.csv and preds_walkforward_cqr.csv")

if __name__=='__main__':
    main()
