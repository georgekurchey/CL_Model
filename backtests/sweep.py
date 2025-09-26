import os, argparse, subprocess as sp, pandas as pd, itertools as it, shutil, sys
from pathlib import Path

def run(cmd):
    r = sp.run(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)
    print(r.stdout)
    if r.returncode != 0:
        sys.exit(r.returncode)

def ensure_dirs():
    Path("preds").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

def sweep_qwalk(mixes, dfs, paths):
    results = []
    for mix, df in it.product(mixes, dfs):
        run([sys.executable, "-m", "backtests.qwalk_backtest", "--mix", str(mix), "--df", str(df), "--paths", str(paths)])
        src = Path("preds/model_qwalk.parquet")
        dst = Path(f"preds/model_qwalk_mix{mix}_df{df}.parquet")
        if src.exists():
            if dst.exists():
                dst.unlink()
            shutil.move(src, dst)
        results.append(("qwalk", mix, df, paths, dst.name))
    return results

def sweep_svjump(ridges):
    out = []
    for r in ridges:
        run([sys.executable, "-m", "backtests.model_backtest", "-r", str(r)])
        src = Path("preds/model_svjump.parquet")
        dst = Path(f"preds/model_svjump_r{r}.parquet")
        if src.exists():
            if dst.exists():
                dst.unlink()
            shutil.move(src, dst)
        out.append(("svjump", r, dst.name))
    return out

def parse_compare(csv_path="reports/compare.csv"):
    df = pd.read_csv(csv_path)
    return df[["model","CRPS","PIT_mean"]].sort_values("CRPS")

def main():
    ensure_dirs()
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixes", default=os.environ.get("QWALK_MIXES","0.35,0.45,0.55"))
    ap.add_argument("--dfs", default=os.environ.get("QWALK_DFS","3,4,6"))
    ap.add_argument("--paths", type=int, default=int(os.environ.get("QWALK_PATHS","6000")))
    ap.add_argument("--ridges", default=os.environ.get("RIDGES","0.00005,0.0001,0.0002"))
    args = ap.parse_args()

    mixes = [float(x) for x in str(args.mixes).split(",") if x]
    dfs = [int(x) for x in str(args.dfs).split(",") if x]
    ridges = [float(x) for x in str(args.ridges).split(",") if x]

    qwalk_runs = sweep_qwalk(mixes, dfs, args.paths)
    sv_runs = sweep_svjump(ridges)

    run([sys.executable, "-m", "backtests.compare"])
    run([sys.executable, "-m", "backtests.coverage"])

    cmp_df = parse_compare()
    cmp_df.to_csv("reports/sweep_compare.csv", index=False)

    best_q = cmp_df[cmp_df["model"].str.startswith("model_qwalk_mix")].head(1)
    best_s = cmp_df[cmp_df["model"].str.startswith("model_svjump_r")].head(1)
    summary = pd.concat([best_q.assign(kind="qwalk"), best_s.assign(kind="svjump")], ignore_index=True)
    summary.to_csv("reports/sweep_best.csv", index=False)
    print("reports/sweep_compare.csv")
    print("reports/sweep_best.csv")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
