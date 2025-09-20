#!/usr/bin/env bash
set -euo pipefail
ROOT="/Users/georgekurchey/CL_Model"
PY="$(command -v python3)"
"$PY" "$ROOT/scripts/validate_cl_strip.py" --path "$ROOT/data/raw/cl_strip.parquet" --expect_m 12 --show 8
