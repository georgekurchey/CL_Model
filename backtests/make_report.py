import re
from pathlib import Path
import numpy as np
import pandas as pd

def pinball(y: np.ndarray, Q: np.ndarray, taus):
    y = np.asarray(y).reshape(-1, 1)
    Q = np.asarray(Q)
    taus = np.asarray(taus).reshape(1, -1)
    return np.maximum(taus * (y - Q), (taus - 1) * (y - Q))

def _eval_one(pred_path: Path, feat: pd.DataFrame):
    pred = pd.read_parquet(pred_path)
    qcols = [c for c in pred.columns if re.fullmatch(r"q\d{2}", c)]
    if not qcols:
        return None
    qcols.sort(key=lambda c: int(c[1:]))

    taus = [int(c[1:]) / 100.0 for c in qcols]

    feat_i = feat.set_index("date")
    pred_i = pred.set_index("date")

    common = feat_i.index.intersection(pred_i.index)
    if common.empty:
        return None

    y = feat_i.loc[common, "ret_1d"].to_numpy()
    Q = pred_i.loc[common, qcols].to_numpy()

    L = pinball(y, Q, taus)
    return pred_path.stem, float(np.nanmean(L))

def main():
    feat_path = Path("data/proc/features.parquet")
    if not feat_path.exists():
        raise SystemExit("features not found; run feature build first")

    feat = pd.read_parquet(feat_path)
    if "date" not in feat.columns or "ret_1d" not in feat.columns:
        raise SystemExit("features missing required columns: date/ret_1d")

    preds_dir = Path("preds")
    results = []
    for f in sorted(preds_dir.glob("*.parquet")):
        r = _eval_one(f, feat)
        if r is not None:
            results.append(r)

    Path("reports").mkdir(parents=True, exist_ok=True)
    out = Path("reports/report.txt")
    if not results:
        out.write_text("No prediction files with quantiles found.\n")
        print(out)
        return

    lines = ["== Pinball loss (avg) =="]
    for name, val in results:
        lines.append(f"{name:20s} {val:.6f}")
    out.write_text("\n".join(lines) + "\n")
    print(out)

if __name__ == "__main__":
    main()
