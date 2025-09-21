venv:
	python3 -m venv .venv
install:
	. .venv/bin/activate && pip install -U pip && pip install ruff pytest pyyaml pandas numpy scikit-learn requests python-dateutil nasdaq-data-link pyarrow fastparquet arch scipy
ingest:
	. .venv/bin/activate && python -m etl.eia --since 2015-01-01 && python -m etl.fred --series DTWEXBGS,DGS10,T10YIE --since 2015-01-01 && python -m etl.nasdaq_chris_settlements || (export SEED_CME=1; python -m etl.nasdaq_chris_settlements)
features:
	. .venv/bin/activate && python -m features.build_features --config config/pipeline.yaml
backtest:
	. .venv/bin/activate && python -m backtests.walkforward
baselines:
	. .venv/bin/activate && python -m backtests.baselines
model_backtest:
	. .venv/bin/activate && python -m backtests.model_backtest
compare:
	. .venv/bin/activate && python -m backtests.compare
coverage:
	. .venv/bin/activate && python -m backtests.coverage
report:
	. .venv/bin/activate && python -m backtests.make_report
test:
	. .venv/bin/activate && pytest -q
