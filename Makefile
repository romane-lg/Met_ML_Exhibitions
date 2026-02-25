.PHONY: setup setup-win lint format type test coverage serve build-features train-ranker evaluate-backends evaluate-comprehensive compare-clip streamlit streamlit-tfidf streamlit-clip

setup:
	mkdir -p .tmp .uv-cache
	TMP=.tmp TEMP=.tmp UV_CACHE_DIR=.uv-cache uv sync --all-extras

setup-win:
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_env.ps1

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

evaluate-comprehensive:
	uv run python -m scripts.evaluate_model_comprehensive --artifacts $${MET_ARTIFACTS_DIR:-artifacts} --top-k 8 --latency-runs 24 --json-out artifacts/eval_comprehensive.json --csv-out artifacts/eval_comprehensive.csv

compare-clip:
	uv run python -m scripts.compare_clip_modes --artifacts $${MET_ARTIFACTS_DIR:-artifacts}
