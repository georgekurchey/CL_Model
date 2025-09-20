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
