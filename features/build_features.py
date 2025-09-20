
import argparse, json, os, math

def business_days(start_year=2016, end_year=2024):
    from datetime import date, timedelta
    d=date(start_year,1,1); end=date(end_year,12,31)
    out=[]
    while d<=end:
        if d.weekday()<5: out.append(d.isoformat())
        d += timedelta(days=1)
    return out

def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header)+"\n")
        for r in rows:
            f.write(",".join(r)+"\n")

def make_demo(out_path):
    dates = business_days(2016, 2024)
    n=len(dates)
    # simple RNG (LCG + Box-Muller)
    seed=42
    def randu():
        nonlocal seed
        seed = (1664525*seed + 1013904223) % (2**32)
        return (seed+1)/(2**32)
    def randn():
        u1=max(randu(),1e-12); u2=randu()
        return ( -2.0*math.log(u1) )**0.5 * math.cos(2*math.pi*u2)

    cl = [[0.0]*n for _ in range(13)]  # 1..12
    usd=[0.0]*n; ust10=[0.0]*n; bei10=[0.0]*n; cvol=[0.0]*n
    eia=[0]*n; inv=[0.0]*n; ret=[0.0]*n
    for i in range(n):
        t=i/252.0
        base=55.0 + 15.0*math.sin(2*math.pi*t/1.0) + 0.8*randn()
        for k in range(1,13):
            cl[k][i] = base + (k-1)*0.05
        usd[i]  = (usd[i-1] if i>0 else 0.0) + 0.02*randn()
        ust10[i]= 2.0 + 0.5*math.sin(2*math.pi*t/1.5) + 0.05*randn()
        bei10[i]= 2.2 + 0.1*math.sin(2*math.pi*t/1.8) + 0.03*randn()
        cvol[i] = 35.0 + 5.0*math.sin(2*math.pi*t/1.1) + 1.2*randn()
        # Wednesday indicator
        y,m,d = [int(p) for p in dates[i].split("-")]
        import datetime as _dt
        eia[i] = 1 if _dt.date(y,m,d).weekday()==2 else 0
        inv[i] = 0.4*randn()*eia[i]

    for i in range(n):
        slope_lag = (cl[1][i-1]-cl[2][i-1]) if i>0 else 0.0
        usd_lag = usd[i-1] if i>0 else 0.0
        iv = inv[i]
        vol = 0.02 + (0.004*(cvol[i-1]-35.0) if i>0 else 0.0)
        vol = max(0.01, min(0.2, vol))
        mean = -0.02*slope_lag - 0.005*usd_lag + 0.015*iv
        ret[i] = mean + vol*randn()

    # rebuild M1 price
    price=55.0
    for i in range(n):
        price *= math.exp(ret[i])
        cl[1][i]=price
        for k in range(2,13):
            cl[k][i]=cl[1][i]+(k-1)*0.05
    spread=[cl[1][i]-cl[2][i] for i in range(n)]

    header=["date"]+[f"cl_settle_m{k}" for k in range(1,13)]+["usd_broad_index","ust_10y","breakeven_10y","wti_cvol_1m","eia_day_dummy","inv_surprise","spread_m1_m2","ret_1d"]
    rows=[]
    for i in range(n):
        row=[dates[i]]+[f"{cl[k][i]:.6f}" for k in range(1,13)]+[f"{usd[i]:.6f}",f"{ust10[i]:.6f}",f"{bei10[i]:.6f}",f"{cvol[i]:.6f}",f"{eia[i]}",
             f"{inv[i]:.6f}",f"{spread[i]:.6f}",f"{ret[i]:.8f}"]
        rows.append(row)
    write_csv(out_path, header, rows)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    cfg = json.load(open(args.config, "r", encoding="utf-8"))
    out_path = os.path.join(cfg["paths"]["proc"], "features.csv")
    if args.demo:
        make_demo(out_path)
        print("Demo features ->", out_path)
    else:
        raise SystemExit("Real-data build not implemented in this minimal Python 3 demo. Use --demo.")
