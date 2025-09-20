#!/usr/bin/env bash
set -euo pipefail
F="/Users/georgekurchey/CL_Model/features/build_features_live.py"
if [ ! -f "$F" ]; then
  echo "Builder not found: $F"
  exit 1
fi
if grep -q "merge_roll_features" "$F"; then
  echo "Roll features already integrated."
  exit 0
fi
BK="$F.bak.$(date +%Y%m%d_%H%M%S)"
cp "$F" "$BK"
python3 - "$F" <<'PYI'
import sys
from pathlib import Path
p = Path(sys.argv[1])
s = p.read_text()
insert_block = '''
# ---- Roll features integration ----
try:
    from features.roll_utils import merge_roll_features
    _strip_path = "/Users/georgekurchey/CL_Model/data/raw/cl_strip.parquet"
    df = merge_roll_features(df, _strip_path, eps=0.01)
    print("Roll features merged: ret_1d_splice, roll_flag, days_since_roll, px_cm1")
except Exception as _e:
    print(f"Roll features skipped: {_e}")
# -----------------------------------
'''
idx = s.rfind('.to_csv(')
if idx == -1:
    s2 = s + "\n" + insert_block + "\n"
else:
    line_start = s.rfind('\n', 0, idx) + 1
    s2 = s[:line_start] + insert_block + s[line_start:]
p.write_text(s2)
print("Patched build_features_live.py (backup created).")
PYI
