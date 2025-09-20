# PowerShell run script
Set-Location $PSScriptRoot
python -m features.build_features --config config/pipeline.json --demo
python -m backtests.walkforward --config config/pipeline.json --folds 5
python -m backtests.scoring --input preds --out reports
Start-Process .\reports\report.html
