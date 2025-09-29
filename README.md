# CL_Model (ready‑to‑run)

- [Getting Started](docs/guide.md)

Unzip this ZIP **directly into your Documents folder** so you get:
    Documents/CL_Model/

## Easiest run

### Windows
- Double‑click: `run_demo.bat`

### macOS / Linux
```bash
chmod +x run_demo.sh
./run_demo.sh
```

## Manual commands (from inside CL_Model/)

**Windows (PowerShell):**
```powershell
python -m features.build_features --config config/pipeline.json --demo
python -m backtests.walkforward --config config/pipeline.json --folds 5
python -m backtests.scoring --input preds --out reports
start reports\report.html
```

**macOS / Linux (Terminal):**
```bash
python3 -m features.build_features --config config/pipeline.json --demo
python3 -m backtests.walkforward --config config/pipeline.json --folds 5
python3 -m backtests.scoring --input preds --out reports
open reports/report.html      # macOS
xdg-open reports/report.html  # Linux
```

## What’s included
- Pure‑Python demo (no internet, no extra packages)
- ARX + EWMA volatility → Normal(μ, σ)
- Linear quantile regression stack (τ=5..95%)
- Walk‑forward: 3y train / 6m test (5 folds)
- Scoring: CRPS + simple VaR coverage
- Output: `reports/report.html`
