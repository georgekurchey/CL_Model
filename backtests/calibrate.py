from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

TAUS = np.arange(0.05,0.96,0.05)
QCOLS = [f"q{int(t*100):02d}" for t in TAUS]

def _calibrate(qs, y, target=(0.05,0.95)):
    q05 = np.quantile(qs, 0.50, axis=1)  # median proxy if not present
    # center each row at its 50th
    med = q05.reshape(-1,1)
    dev = qs - med
    def cov_at(dev, med, y, a):
        # a=scale (temperature)
        q_adj = med + a*dev
        # coverage at nominal 5% and 95% cols
        i05 = QCOLS.index("q05"); i95 = QCOLS.index("q95")
        c05 = np.mean(y < q_adj[:,i05])
        c95 = np.mean(y < q_adj[:,i95])
        return c05, c95, q_adj
    # grid search a∈[0.8, 2.0]
    best = (1e9, 1.0, None, 0.0, 0.0)
    for a in np.linspace(0.8, 2.0, 25):
        c05,c95,q_adj = cov_at(dev, med, y, a)
        loss = (c05-target[0])**2 + (c95-target[1])**2
        if loss < best[0]:
            best = (loss, a, q_adj, c05, c95)
    return {"a":best[1], "q":best[2], "c05":best[3], "c95":best[4]}

def main(pred_path="preds/model_qwalk.parquet", feat_path="data/proc/features.parquet",
         out_path="preds/model_qwalk_cal.parquet"):
    dfp = pd.read_parquet(pred_path)
    feat = pd.read_parquet(feat_path)[["date","ret_1d"]]
    df = dfp.merge(feat, on="date", how="inner")
    y = df["ret_1d"].to_numpy()
    qs = df[QCOLS].to_numpy()
    res = _calibrate(qs, y)
    qadj = res["q"]
    out = pd.DataFrame({"date": df["date"]})
    for i,t in enumerate(TAUS):
        out[f"q{int(t*100):02d}"] = qadj[:,i]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"{out_path} a={res['a']:.3f} cov05={res['c05']:.3f} cov95={res['c95']:.3f}")

if __name__ == "__main__":
    main()
