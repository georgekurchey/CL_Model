#!/usr/bin/env bash
set -euo pipefail
ROOT="/Users/georgekurchey/CL_Model"
[ -f "$ROOT/.venv/bin/activate" ] && . "$ROOT/.venv/bin/activate" || true
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

python3 - <<'PY2'
import pandas as pd
from pathlib import Path
from features.roll_utils import merge_roll_features

feat = Path("/Users/georgekurchey/CL_Model/data/proc/features.csv")
strip = Path("/Users/georgekurchey/CL_Model/data/raw/cl_strip.parquet")

df = pd.read_csv(feat, parse_dates=["date"])
df2 = merge_roll_features(df, str(strip), eps=0.01)
df2.to_csv(feat, index=False)
print("Post-processed features with roll features ->", feat)

# quick sanity print
small = df2.tail(3)[["date","roll_flag","ret_1d_splice","px_cm1","days_since_roll"]]
print(small.to_string(index=False))
PY2
