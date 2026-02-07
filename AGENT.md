# AGENT.md

## Purpose
Repository playbook for contributors and automation agents.

## Project Scope
Build and serve themed artwork recommendations using merged image and metadata text features.

## Source Layout
- `src/app`: Streamlit application.
- `src/api`: FastAPI service.
- `src/data`: dataset load and validation helpers.
- `src/features`: image and text feature extraction modules.
- `src/models`: recommender and scoring logic.
- `scripts`: build/train entrypoints.
- `tests`: unit tests.

## Required Workflow
1. Create branch from `develop`: `feature/<name>`.
2. Keep commits modular and focused.
3. Run `make lint test type` before PR.
4. Merge feature branch into `develop` via PR.
5. Use `release/<version>` for releases.

## Tooling
- Dependency management: `uv`.
- Lint/format: `ruff`.
- Type check: `ty`.
- Tests/coverage: `pytest`, `pytest-cov`.
- Load tests: `locust`.

## Runtime Commands
- `make build-features`
- `make train-ranker`
- `make serve`
- `make streamlit`
- Windows/VS Code alternative (recommended for this repo):
  - use `.vscode/launch.json` and `.vscode/tasks.json` one-click entries.
  - direct equivalents:
    - `python -m uv run python scripts/build_features.py --force`
    - `python -m uv run python scripts/train_ranker.py`
    - `python -m uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000`
    - `python -m uv run streamlit run src/app/streamlit_app.py --server.port 8501`

## VS Code One-Click Run
- Shared run profiles are in `.vscode/launch.json`.
- Use Run/Debug dropdown for:
  - `Build Features`
  - `Train Ranker`
  - `Run Streamlit App`
  - `Run API (Uvicorn)`
  - `Run API + Streamlit`
- Shared command tasks are in `.vscode/tasks.json`:
  - restart API/Streamlit
  - build features
  - train ranker
  - run tests
- First-time setup (required before one-click runs):
  - `python -m uv sync --all-extras`
- Launch configs are pinned to project interpreter:
  - `${workspaceFolder}\\.venv\\Scripts\\python.exe`

## Environment
Use `.env` with:
- `GOOGLE_APPLICATION_CREDENTIALS`
- `GOOGLE_CLOUD_PROJECT`
- `MET_DATA_CSV`
- `MET_IMAGES_DIR`
- `MET_ARTIFACTS_DIR`
- `MET_ENABLE_VISION`

## Consumer vs Maintainer
- Consumer mode:
  - `MET_ENABLE_VISION=false`
  - run app from prebuilt artifacts; no API key required.
- Maintainer mode:
  - `MET_ENABLE_VISION=true`
  - valid `GOOGLE_APPLICATION_CREDENTIALS` required
  - can rebuild artifacts and retrain model.

## Guardrails
- Always follow repository standards in `docs/BEST_PRACTICES.md`.
- Any code change that affects behavior, interfaces, commands, configuration, or outputs must be accompanied by matching documentation updates in `README.md` and/or `docs/*`.
- Never commit secrets.
- Do not rewrite history on shared branches.
- Avoid destructive git commands unless explicitly requested.
- Keep data bootstrap commit as baseline; avoid future raw-data churn.
