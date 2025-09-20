
import argparse, math
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

ROOT = Path("/Users/georgekurchey/CL_Model")

def parse_taus(s):
    if s:
        vals=[]
        for tok in s.split(","):
            tok=tok.strip()
            if not tok: continue
            if tok.endswith("%"): vals.append(float(tok[:-1])/100.0)
            elif float(tok)>1.0:  vals.append(float(tok)/100.0)
            else:                 vals.append(float(tok))
        return sorted([t for t in vals if 0.0<t<1.0])
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

def fit_quantile_models(X,y,taus,n_estimators=600,max_depth=3,learning_rate=0.05,min_samples_leaf=80,random_state=42):
    models={}
    for t in taus:
        est=GradientBoostingRegressor(loss="quantile",alpha=t,
                                      n_estimators=n_estimators,max_depth=max_depth,
                                      learning_rate=learning_rate,min_samples_leaf=min_samples_leaf,
                                      random_state=random_state)
        est.fit(X,y)
        models[t]=est
    return models

def predict_grid(models,X,taus):
    preds=np.column_stack([models[t].predict(X) for t in taus]) if len(taus)>0 else np.zeros((len(X),0))
    if preds.size: preds=np.maximum.accumulate(preds,axis=1)  # enforce monotone in tau
    return preds

def calib_curve_from_cal_window(y_cal,q_cal,taus):
    cov=[float(np.mean(y_cal<=q_cal[:,i])) for i,_ in enumerate(taus)]
    iso=IsotonicRegression(y_min=0.0,y_max=1.0,increasing=True,out_of_bounds="clip")
    x=np.array(taus,float); y=np.array(cov,float)
    h=iso.fit(x,y)
    xs=h.predict(x); ys=x
    for i in range(1,len(xs)):
        if xs[i]<=xs[i-1]: xs[i]=xs[i-1]+1e-9
    return xs,ys   # invert by interpolating (xs->ys)

def apply_calibration(q_pred,taus,xs_cov,ys_tau,targets):
    taus=np.array(taus,float); xs=np.array(xs_cov,float); ys=np.array(ys_tau,float)
    targets=np.array(targets,float)
    tau_eff=np.interp(targets,xs,ys)
    out=np.empty((q_pred.shape[0],len(targets)),float)
    for i in range(q_pred.shape[0]):
        out[i,:]=np.interp(tau_eff,taus,q_pred[i,:])
    return out

def main():
    ap=argparse.ArgumentParser(description="Walk-forward (gbt_iso): quantile GBT + isotonic level calibration.")
    ap.add_argument("--features",default=str(ROOT/"data/proc/features.csv"))
    ap.add_argument("--outdir",default=str(ROOT/"preds"))
    ap.add_argument("--report",default=str(ROOT/"reports/wf_report.html"))
    ap.add_argument("--folds",type=int,default=5)
    ap.add_argument("--val_days",type=int,default=126)
    ap.add_argument("--min_train_days",type=int,default=504)
    ap.add_argument("--iso_days",type=int,default=252)
    ap.add_argument("--taus",default="5,10,20,30,40,50,60,70,80,90,95")
    ap.add_argument("--n_estimators",type=int,default=600)
    ap.add_argument("--max_depth",type=int,default=3)
    ap.add_argument("--learning_rate",type=float,default=0.05)
    ap.add_argument("--min_samples_leaf",type=int,default=80)
    args=ap.parse_args()

    taus=parse_taus(args.taus)
    df=pd.read_csv(args.features,parse_dates=["date"]).sort_values("date")
    if "ret_1d" not in df.columns: raise SystemExit("features must contain 'ret_1d' as target.")
    feat_cols=[c for c in df.columns if c not in ("date","ret_1d")]
    Xall=df[feat_cols].astype(float).fillna(0.0).values
    yall=df["ret_1d"].astype(float).values

    n=len(df); n_val_total=args.folds*args.val_days
    train_end0=n-n_val_total
    if train_end0<args.min_train_days:
        args.val_days=max(30,(n-args.min_train_days)//max(1,args.folds))
        n_val_total=args.folds*args.val_days; train_end0=n-n_val_total
        if train_end0<args.min_train_days: raise SystemExit("Not enough rows for folds/min_train_days/val_days.")

    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    all_preds=[]; fold_metrics=[]
    for k in range(args.folds):
        tr_end=train_end0+k*args.val_days; va_end=tr_end+args.val_days
        tr_idx=slice(0,tr_end); va_idx=slice(tr_end,va_end)
        ntr=tr_end; ncal=min(args.iso_days,ntr); cal_idx=slice(tr_end-ncal,tr_end)

        Xtr,ytr=Xall[tr_idx],yall[tr_idx]
        Xcal,ycal=Xall[cal_idx],yall[cal_idx]
        Xva,yva=Xall[va_idx],yall[va_idx]
        d_va=df["date"].iloc[va_idx]

        models=fit_quantile_models(Xtr,ytr,taus,
                                   n_estimators=args.n_estimators,max_depth=args.max_depth,
                                   learning_rate=args.learning_rate,min_samples_leaf=args.min_samples_leaf,
                                   random_state=42+k)
        q_cal_raw=predict_grid(models,Xcal,taus)
        xs_cov,ys_tau=calib_curve_from_cal_window(ycal,q_cal_raw,taus)

        q_va_raw=predict_grid(models,Xva,taus)
        q_va_cal=apply_calibration(q_va_raw,taus,xs_cov,ys_tau,taus)

        crps=crps_from_grid(yva,q_va_cal,taus)
        qnames=[f"q{int(t*100):02d}" for t in taus]
        df_fold=pd.DataFrame({"date":d_va.values,"y":yva})
        for i,name in enumerate(qnames): df_fold[name]=q_va_cal[:,i]
        var5=int((df_fold["y"]<df_fold["q05"]).sum()) if "q05" in df_fold.columns else None
        var95=int((df_fold["y"]>df_fold["q95"]).sum()) if "q95" in df_fold.columns else None
        df_fold["crps"]=crps
        df_fold.to_csv(outdir/f"preds_fold{k+1}.csv",index=False)
        all_preds.append(df_fold)
        crps_mean=float(np.mean(crps)) if len(crps)>0 else float("nan")
        print(f"[fold {k+1}] n_val={len(df_fold)} CRPS_mean={crps_mean:.6f} var5={var5} var95={var95}")
        fold_metrics.append({"fold":k+1,"n_val":int(len(df_fold)),"crps_mean":crps_mean,"var5_exceed":var5,"var95_exceed":var95})

    agg=pd.concat(all_preds,ignore_index=True); agg.to_csv(outdir/"preds_walkforward.csv",index=False)
    ov={"n_total":int(len(agg)),
        "crps_mean":float(np.mean(agg["crps"])) if len(agg)>0 else float("nan"),
        "var5_exceed": int((agg["y"]<agg["q05"]).sum()) if "q05" in agg.columns else None,
        "var95_exceed": int((agg["y"]>agg["q95"]).sum()) if "q95" in agg.columns else None}

    html=[]
    html.append("<html><head><meta charset='utf-8'><title>Walk-Forward Report</title>")
    html.append("<style>body{font-family:Arial,Helvetica,sans-serif;margin:20px;} table{border-collapse:collapse;} th,td{border:1px solid #ddd;padding:6px;} th{background:#f0f0f0;}</style></head><body>")
    html.append("<h2>CL_Model — Walk-Forward (gbt_iso)</h2>")
    html.append(f"<p>Folds={args.folds} | val_days={args.val_days} | iso_days={args.iso_days} | taus={','.join([str(int(t*100)) for t in taus])}</p>")
    html.append("<h3>Per-fold metrics</h3>"+pd.DataFrame(fold_metrics).to_html(index=False))
    html.append("<h3>Overall</h3>"+pd.DataFrame([ov]).to_html(index=False))
    tail=agg.tail(15).copy(); tail["date"]=pd.to_datetime(tail["date"]).dt.strftime("%Y-%m-%d")
    html.append("<h3>Last 15 predictions</h3>"+tail.to_html(index=False))
    html.append("</body></html>")
    Path(args.report).write_text("\n".join(html),encoding="utf-8")
    print(f"[report] -> {args.report}")
    print(f"[wf] preds in {outdir}/preds_fold*.csv and preds_walkforward.csv")

if __name__=="__main__": main()
