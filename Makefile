.PHONY: install check lint format-check format test ci mlflow-ui app

PYTHON_SOURCES = src tests notebooks

install:
	poetry install

check:
	poetry check --strict
	$(MAKE) lint
	$(MAKE) format-check

lint:
	poetry run ruff check $(PYTHON_SOURCES)

format-check:
	poetry run ruff format --check $(PYTHON_SOURCES)

format:
	poetry run ruff format $(PYTHON_SOURCES)

test:
	CONTEXTIQ_LOG_CONTEXT=tests CONTEXTIQ_LOG_CONSOLE=0 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 poetry run python -m unittest discover -s tests -p 'test_*.py' -v

ci: check test

mlflow-ui:
	MLFLOW_DISABLE_TELEMETRY=true poetry run python -m contextiq.monitoring.mlflow_tracking

app:
	MLFLOW_DISABLE_TELEMETRY=true poetry run streamlit run src/contextiq/ui/app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
