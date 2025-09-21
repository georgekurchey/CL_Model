.PHONY: venv install fmt lint test ingest features backtest report
venv:
	python3 -m venv .venv
install:
	. .venv/bin/activate && \
	pip install -U pip && \
	pip install ruff pytest pyyaml pandas numpy scikit-learn requests python-dateutil
fmt:
	. .venv/bin/activate && ruff format .
lint:
	. .venv/bin/activate && ruff check .
test:
	. .venv/bin/activate && pytest -q
ingest:
	. .venv/bin/activate && python -m etl.eia --since 2015-01-01 && python -m etl.fred --series DTWEXBGS,DGS10,T10YIE --since 2015-01-01
features:
	. .venv/bin/activate && python -m features.build_features --config config/pipeline.yaml
backtest:
	. .venv/bin/activate && python -m backtests.walkforward --config config/pipeline.yaml
report:
	. .venv/bin/activate && python -m backtests.scoring --out reports
