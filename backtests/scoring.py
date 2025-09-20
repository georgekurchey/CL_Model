import argparse, os, csv, math

def read_preds(folder):
    files=[fn for fn in os.listdir(folder) if fn.startswith("preds_fold") and fn.endswith(".csv")]
    files.sort()
    rows=[]
    for fn in files:
        with open(os.path.join(folder, fn), "r", encoding="utf-8") as f:
            r=csv.DictReader(f); rows.extend(list(r))
    return rows

def crps_normal(y, mu, sigma):
    sigma = sigma if sigma>0 else 1e-6
    z=(y-mu)/sigma
    phi=(1.0/(2*math.pi)**0.5)*math.exp(-0.5*z*z)
    Phi=0.5*(1.0+math.erf(z/(2**0.5)))
    return sigma*( z*(2*Phi-1) + 2*phi - 1.0/(math.pi**0.5) )

def approx_crps_from_quantiles(y, qs, taus):
    pin=[]
    for i,q in enumerate(qs):
        diff=y-q
        tau=taus[i]
        pin.append((tau - (1.0 if diff<0 else 0.0))*diff)
    area=0.0
    for i in range(1,len(taus)):
        area += (pin[i-1]+pin[i])*(taus[i]-taus[i-1])/2.0
    return 2.0*area

def binom_p_two_sided(x, n, p=0.05):
    if n==0: return 1.0
    mu=n*p; var=n*p*(1-p); sd=var**0.5 if var>0 else 1.0
    z=abs(x-mu)/sd if sd>0 else 0.0
    Phi=0.5*(1.0+math.erf(z/(2**0.5)))
    return max(0.0, min(1.0, 2.0*(1.0-Phi)))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    a=ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows=read_preds(a.input)
    models={}
    for r in rows:
        models.setdefault(r["model"], []).append(r)
    taus=[i/100.0 for i in range(5,100,5)]
    html=["<html><head><meta charset='utf-8'></head><body><h1>Stage-2 Backtest Report (Pure Python 3)</h1>"]
    for m,rs in models.items():
        crps=[]; var5=0; var95=0; n=0
        for r in rs:
            y=float(r["y"])
            if m=="arx_ewma":
                mu=float(r["mu"]); sigma=float(r["sigma"] or 0.0)
                crps.append(crps_normal(y, mu, sigma))
            else:
                qs=[float(r[f"q{int(t*100):02d}"]) for t in taus]
                crps.append(approx_crps_from_quantiles(y, qs, taus))
            q05=float(r.get("q05") or "nan"); q95=float(r.get("q95") or "nan")
            if q05==q05 and y<q05: var5+=1
            if q95==q95 and y>q95: var95+=1
            n+=1
        mean_crps = sum(crps)/max(1,len(crps))
        p05 = binom_p_two_sided(var5, n, 0.05)
        p95 = binom_p_two_sided(var95, n, 0.05)
        html.append(f"<h2>Model: {m}</h2>")
        html.append(f"<p><b>Mean CRPS:</b> {mean_crps:.6f}</p>")
        html.append(f"<p><b>VaR 5% exceedances:</b> {var5}/{n} (approx p={p05:.3f})</p>")
        html.append(f"<p><b>VaR 95% exceedances:</b> {var95}/{n} (approx p={p95:.3f})</p>")
    html.append("</body></html>")
    with open(os.path.join(a.out,"report.html"),"w",encoding="utf-8") as f:
        f.write("\n".join(html))
    print("Report ->", os.path.join(a.out,"report.html"))

  
