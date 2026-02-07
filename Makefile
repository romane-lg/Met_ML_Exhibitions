.PHONY: setup lint format type test coverage serve build-features train-ranker streamlit

setup:
	python -m uv sync --all-extras

lint:
	python -m uv run ruff check .

format:
	python -m uv run ruff format .

type:
	python -m uv run ty check .

test:
	python -m uv run pytest -q

coverage:
	python -m uv run pytest --cov=src --cov=scripts --cov-report=term-missing --cov-report=html

serve:
	python -m uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

streamlit:
	python -m uv run streamlit run src/app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0

build-features:
	python -m uv run python scripts/build_features.py

train-ranker:
	python -m uv run python scripts/train_ranker.py
