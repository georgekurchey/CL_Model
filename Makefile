.PHONY: install ingest ingest_cftc ingest_cvol features baselines model_backtest qwalk_backtest qwalk_calibrate compare coverage pitplots report test

RIDGE?=0.0001
QWALK_MIX?=0.45
QWALK_DF?=4
QWALK_PATHS?=8000

install:
	. .venv/bin/activate && pip install -U pip && pip install ruff pytest pyyaml pandas numpy scikit-learn requests python-dateutil nasdaq-data-link pyarrow fastparquet arch scipy matplotlib

ingest:
	. .venv/bin/activate && python -m etl.eia --since 2015-01-01 && python -m etl.fred --series DTWEXBGS,DGS10,T10YIE --since 2015-01-01

ingest_cftc:
	. .venv/bin/activate && python -m etl.cftc

ingest_cvol:
	. .venv/bin/activate && python -m etl.cvol

features:
	. .venv/bin/activate && python -m features.build_features --config config/pipeline.yaml

baselines:
	. .venv/bin/activate && python -m backtests.baselines

model_backtest:
	. .venv/bin/activate && python -m backtests.model_backtest -r $(RIDGE)

qwalk_backtest:
	. .venv/bin/activate && python -m backtests.qwalk_backtest --mix $(QWALK_MIX) --df $(QWALK_DF) --paths $(QWALK_PATHS)

qwalk_calibrate:
	. .venv/bin/activate && python -m backtests.calibrate

compare:
	. .venv/bin/activate && python -m backtests.compare

coverage:
	. .venv/bin/activate && python -m backtests.coverage

pitplots:
	. .venv/bin/activate && python -m backtests.plots

report:
	. .venv/bin/activate && python -m backtests.make_report

test:
	. .venv/bin/activate && pytest -q

stage6_all:
	. .venv/bin/activate && python -m backtests.model_backtest -r $(RIDGE)
	. .venv/bin/activate && python -m backtests.qwalk_backtest --mix $(QWALK_MIX) --df $(QWALK_DF) --paths $(QWALK_PATHS)
	. .venv/bin/activate && python -m backtests.calibrate
	. .venv/bin/activate && python -m backtests.compare
	. .venv/bin/activate && python -m backtests.coverage
	. .venv/bin/activate && python -m backtests.plots
	. .venv/bin/activate && python -m backtests.make_report


daily:
	. .venv/bin/activate && python -m etl.eia --since 2015-01-01
	. .venv/bin/activate && python -m etl.fred --series DTWEXBGS,DGS10,T10YIE --since 2015-01-01
	. .venv/bin/activate && python -m etl.cftc || true
	. .venv/bin/activate && ( [ -f data/raw/cvol/wti_cvol.csv ] \
		&& python -m etl.cvol --csv data/raw/cvol/wti_cvol.csv \
		|| python -m etl.cvol --proxy-ovx )
	. .venv/bin/activate && python -m etl.nasdaq_chris_settlements || true
	make features && make stage6_all
	mkdir -p reports/archive && cp reports/report.txt reports/archive/report_$$(date +%F).txt || true


.PHONY: healthcheck
healthcheck:
	. .venv/bin/activate && python -m backtests.healthcheck

.PHONY: ci_seed
ci_seed:
	. .venv/bin/activate && python -m etl.ci_seed

.PHONY: ci_pipeline
ci_pipeline: ci_seed
	. .venv/bin/activate && python -m features.build_features --config config/pipeline.yaml
	. .venv/bin/activate && python -m backtests.model_backtest -r 0.0001
	. .venv/bin/activate && python -m backtests.qwalk_backtest --mix 0.45 --df 4 --paths 2000
	. .venv/bin/activate && python -m backtests.calibrate
	. .venv/bin/activate && python -m backtests.compare
	. .venv/bin/activate && python -m backtests.coverage
	. .venv/bin/activate && python -m backtests.make_report
	. .venv/bin/activate && python -m backtests.healthcheck

.PHONY: all_daily
all_daily:
	$(MAKE) daily || true
