@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
  set "PY=py -3"
) else (
  set "PY=python"
)
%PY% -m features.build_features --config config/pipeline.json --demo
%PY% -m backtests.walkforward --config config/pipeline.json --folds 5
%PY% -m backtests.scoring --input preds --out reports
start "" "%cd%\reports\report.html"
echo.
echo Done. Press any key to exit.
pause >nul
endlocal
