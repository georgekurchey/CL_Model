.PHONY: venv install fmt lint test precommit

venv:
	python3 -m venv .venv

install:
	. .venv/bin/activate && \
	pip install -U pip && \
	( [ -f requirements.txt ] && pip install -r requirements.txt || true ) && \
	pip install ruff pytest pre-commit

fmt:
	. .venv/bin/activate && ruff format .

lint:
	. .venv/bin/activate && ruff check .

test:
	. .venv/bin/activate && pytest -q

precommit:
	. .venv/bin/activate && pre-commit install
