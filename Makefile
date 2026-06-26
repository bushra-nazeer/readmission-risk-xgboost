.PHONY: install data train evaluate explain card serve test lint format clean

install:
	uv venv --python 3.12
	uv pip install -e ".[dev]"

data:
	uv run python -m readmission.data

train:
	uv run python -m readmission.train --model xgboost

evaluate:
	uv run python -m readmission.evaluate

explain:
	uv run python -m readmission.explain

card:
	uv run python -m readmission.model_card

serve:
	uv run uvicorn readmission.api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

clean:
	rm -rf data/raw data/processed mlruns reports/figures .pytest_cache .ruff_cache
