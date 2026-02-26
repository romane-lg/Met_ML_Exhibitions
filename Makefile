.PHONY: run setup clean lfs-pull lint format type test coverage serve build-features train-ranker evaluate-backends evaluate-comprehensive compare-clip streamlit streamlit-tfidf docker-up docker-down

export PYTHONPATH := .
# Required on macOS: prevent segfault from conflicting OpenMP runtimes (PyTorch vs scikit-learn/xgboost)
export KMP_DUPLICATE_LIB_OK := TRUE
# Required on macOS: prevent segfault when Streamlit forks a process after PyTorch/ObjC runtime is initialized
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY := YES

# Run the full pipeline from scratch: install deps → build features → train → launch UI
run: setup build-features train-ranker streamlit

setup:
	uv sync --all-extras
	uv run python -c "import sklearn, xgboost, psutil, PIL, open_clip, torch; print('OK: sklearn', sklearn.__version__, '| xgboost', xgboost.__version__, '| psutil', psutil.__version__, '| PIL', PIL.__version__)"

clean:
	-Remove-Item -Recurse -Force .venv 2>NUL || rmdir /s /q .venv 2>NUL || rm -rf .venv
	uv sync --all-extras
	uv run python -c "import sklearn, xgboost, psutil, PIL, open_clip, torch; print('OK: sklearn', sklearn.__version__, '| xgboost', xgboost.__version__, '| psutil', psutil.__version__, '| PIL', PIL.__version__)"

lfs-pull:
	git lfs pull

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
ifeq ($(OS),Windows_NT)
	cmd /c "set OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES&& set KMP_DUPLICATE_LIB_OK=TRUE&& set OMP_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& uv run streamlit run src/app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.fileWatcherType none"
else
	OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run streamlit run src/app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.fileWatcherType none
endif

streamlit-tfidf:
	cmd /c "set MET_ARTIFACTS_DIR=artifacts_tfidf&& set MET_AUTO_BUILD_ON_STARTUP=false&& set MET_ENABLE_VISION=false&& uv run streamlit run src/app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0"

build-features:
	uv run python -m scripts.build_features

train-ranker:
	uv run python -m scripts.train_ranker


evaluate-comprehensive:
	uv run python -m scripts.evaluate_model_comprehensive --artifacts $${MET_ARTIFACTS_DIR:-artifacts} --top-k 8 --latency-runs 24 --json-out artifacts/eval_comprehensive.json --csv-out artifacts/eval_comprehensive.csv

compare-clip:
	uv run python -m scripts.compare_clip_modes --artifacts $${MET_ARTIFACTS_DIR:-artifacts}

docker-up:
	@if not exist .env copy .env.example .env
	docker compose up --build

docker-down:
	docker compose down
