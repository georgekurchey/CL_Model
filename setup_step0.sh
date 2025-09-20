#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/georgekurchey/CL_Model"
cd "$ROOT"

# 1) Folders
mkdir -p \
  config \
  data/raw \
  data/proc \
  data/tmp \
  preds \
  reports \
  logs \
  ingest \
  scripts

# Ensure Python packages resolve
mkdir -p backtests models features
[ -f backtests/__init__.py ] || : > backtests/__init__.py
[ -f models/__init__.py ]   || : > models/__init__.py
[ -f features/__init__.py ] || : > features/__init__.py
[ -f ingest/__init__.py ]   || : > ingest/__init__.py

# 2) Secrets template (API keys)
cat > config/secrets.env.example <<'ENV'
# Copy this file to config/secrets.env and fill in your keys.
export NASDAQ_DATA_LINK_API_KEY=CHANGE_ME
export FRED_API_KEY=CHANGE_ME
export EIA_API_KEY=CHANGE_ME
ENV

# Make a working secrets file if it doesn't exist
if [ ! -f config/secrets.env ]; then
  cp config/secrets.env.example config/secrets.env
fi

# 3) Default pipeline config (only create if missing)
if [ ! -f config/pipeline.json ]; then
  cat > config/pipeline.json <<'JSON'
{
  "paths": {
    "root": "/Users/georgekurchey/CL_Model",
    "raw": "data/raw",
    "proc": "data/proc",
    "preds": "preds",
    "reports": "reports",
    "logs": "logs"
  },
  "features": {
    "use_ovx": false,
    "realized_vol_lambda": 0.94,
    "realized_vol_window": 20,
    "forward_fill_days_macro": 1
  },
  "walkforward": {
    "train_years": 3,
    "test_months": 6
  }
}
JSON
fi

# 4) .gitignore (append safe defaults)
{
  echo "# === CL_Model defaults ==="
  echo "config/secrets.env"
  echo "config/secrets.env.example"
  echo "data/raw/"
  echo "data/proc/"
  echo "data/tmp/"
  echo "preds/"
  echo "reports/"
  echo "logs/"
  echo "*.parquet"
  echo "*.csv.gz"
  echo ".DS_Store"
  echo ".venv/"
} >> .gitignore 2>/dev/null || true

# 5) Simple README note (create if absent)
if [ ! -f README_LIVE_DATA.md ]; then
  cat > README_LIVE_DATA.md <<'MD'
# Live Data Setup (Step 0)

1) Put your API keys into `config/secrets.env`:
   - NASDAQ_DATA_LINK_API_KEY
   - FRED_API_KEY
   - EIA_API_KEY

2) Folder roles:
   - `data/raw/`  : raw API pulls (parquet)
   - `data/proc/` : cleaned/merged features
   - `preds/`     : model predictions
   - `reports/`   : HTML/CSV reports
   - `logs/`      : ingest logs

3) Next: implement ingest scripts for
   - futures strip (Nasdaq Data Link)
   - macro (FRED)
   - EIA weekly
   - realized vol (fallback if OVX unavailable)
MD
fi

echo "Step 0 complete."
echo "Now edit your API keys in: $ROOT/config/secrets.env"
