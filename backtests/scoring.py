from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd

def save_dummy_report(preds: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    html = out_dir / "report.html"
    head = "<h1>CL_Model Report (stub)</h1>"
    body = preds.tail(10).to_html(index=False)
    html.write_text(head + body)
    return html

def main(out: str) -> None:
    f = Path("preds/baseline_quantiles.parquet")
    if not f.exists():
        raise SystemExit("missing preds/baseline_quantiles.parquet")
    df = pd.read_parquet(f)
    p = save_dummy_report(df, Path(out))
    print(str(p))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    main(args.out)
