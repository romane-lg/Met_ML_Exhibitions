.PHONY: setup lint format type test coverage serve build-features train-ranker evaluate-backends streamlit streamlit-tfidf streamlit-clip

setup:
	uv sync --all-extras

lint:
	uv run ruff check .

format:
	uv run ruff format .

type:
	uv run ty check src scripts tests

test:
	uv run pytest -q

coverage:
	uv run pytest --cov=src --cov=scripts --cov-report=term-missing --cov-report=html

serve:
	uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

streamlit:
	PYTHONPATH=. uv run streamlit run src/app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0

streamlit-tfidf:
	PYTHONPATH=. MET_ARTIFACTS_DIR=artifacts_tfidf MET_AUTO_BUILD_ON_STARTUP=false MET_ENABLE_VISION=false uv run streamlit run src/app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0

streamlit-clip:
	PYTHONPATH=. MET_ARTIFACTS_DIR=artifacts_clip MET_AUTO_BUILD_ON_STARTUP=false MET_ENABLE_VISION=false uv run streamlit run src/app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0

build-features:
	uv run python -m scripts.build_features

train-ranker:
	uv run python -m scripts.train_ranker

evaluate-backends:
	uv run python -m scripts.evaluate_backends --artifacts artifacts_tfidf artifacts_clip --k 10
