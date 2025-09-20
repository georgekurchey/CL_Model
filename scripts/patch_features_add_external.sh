#!/usr/bin/env bash
set -euo pipefail
F="/Users/georgekurchey/CL_Model/features/build_features_live.py"
if [ ! -f "$F" ]; then echo "Builder not found: $F"; exit 1; fi
if grep -q "merge_spread_features" "$F" && grep -q "cot.csv" "$F" && grep -q "fred_macro.csv" "$F"; then
  echo "External merges already integrated."
  exit 0
fi
BK="$F.bak.$(date +%Y%m%d_%H%M%S)"
cp "$F" "$BK"

python3 - "$F" <<'PYI'
import sys
from pathlib import Path
p = Path(sys.argv[1])
s = p.read_text()

block = '''
# ---- External merges: Brent spread, CFTC COT, FRED macro ----
try:
    from pathlib import Path
    import pandas as pd
    try:
        from features.spread_features import merge_spread_features
        df = merge_spread_features(df,
            brent_csv="/Users/georgekurchey/CL_Model/data/external/brent.csv",
            wti_price_col="px_cm1",
            lookback=252)
        print("Spread features merged: wti_brent_spread, wti_brent_z, brent_close")
    except Exception as _e:
        print(f"Spread features skipped: {_e}")

    try:
        cot_p = Path("/Users/georgekurchey/CL_Model/data/external/cot.csv")
        if cot_p.exists():
            cot = pd.read_csv(cot_p, parse_dates=["date"])
            df = df.merge(cot, on="date", how="left")
            print("CFTC COT merged (cols:", list(cot.columns), ")")
        else:
            print("CFTC COT file not found; skipping.")
    except Exception as _e:
        print(f"CFTC merge skipped: {_e}")

    try:
        macro_p = Path("/Users/georgekurchey/CL_Model/data/external/fred_macro.csv")
        if macro_p.exists():
            macro = pd.read_csv(macro_p, parse_dates=["date"])
            df = df.merge(macro, on="date", how="left")
            print("FRED macro merged (cols:", list(macro.columns), ")")
        else:
            print("FRED macro file not found; skipping.")
    except Exception as _e:
        print(f"FRED macro skipped: {_e}")
except Exception as _e:
    print(f"External merges block failed: {_e}")
# -------------------------------------------------------------
'''

# insert just before the final to_csv
idx = s.rfind('.to_csv(')
if idx == -1:
    s2 = s + "\n" + block + "\n"
else:
    line_start = s.rfind('\n', 0, idx) + 1
    s2 = s[:line_start] + block + s[line_start:]

p.write_text(s2)
print("Patched build_features_live.py (backup created).")
PYI
