# Met_ML_Exhibitions

Automated recommendation system for themed MET exhibitions using combined image and metadata text signals.

## What Changed
- Added engineering tooling with `uv`, `ruff`, `ty`, and standardized `make` targets.
- Added API service (`FastAPI`) and containerization (`Docker`, `docker-compose`).
- Reworked Streamlit to use real artifacts with theme filters, image display, and similarity thresholding.
- Added feature build/train scripts with artifact caching and optional Google Vision integration.
- Added test + coverage + load-test scaffolding.

## Why Makefile
- Single command surface for all common tasks.
- Reduces setup drift across different machines.
- Improves onboarding and CI consistency.
- Keeps commands explicit and reviewable in one place (`Makefile`).

## Extended Docs
- Methodology and design decisions: `docs/METHODOLOGY_AND_DECISIONS.md`
- CLIP migration summary: `docs/CLIP_MIGRATION_SUMMARY.md`
- General best practices: `docs/BEST_PRACTICES.md`

## Architecture
- `src/loaders/image_api_loader.py`: Google Vision API loader (request/response + retries/errors), no cleaning.
- `src/features/image_features.py`: cleans and tokenizes raw Vision response into feature-ready image signals.
- `src/features/nlp_utils.py`: shared NLP tokenization/lemmatization utilities for text and image-derived tokens.
- `scripts/build_features.py`: orchestrates data loading + image/text feature modules and writes artifacts.
- `scripts/train_ranker.py`: trains optional LightGBM ranker on embedding-difference + cosine + numeric-feature differences.
- `src/models/recommender.py`: Theme recommendation, exhibition grouping, coherence scoring.
- `src/api/main.py`, `src/api/routes.py`: FastAPI endpoints.
- `src/app/streamlit_app.py`: Interactive curator UI with image previews and filters.

## Data Layout
- `data/raw/met_data.csv`: metadata table.
- `data/raw/images/`: image files referenced by `image_path`.
- `artifacts/`: generated assets.
- `artifacts/embeddings.npz`
- `artifacts/combined_embeddings.pkl`
- `artifacts/meta.csv`
- `artifacts/tokens.json`
- `artifacts/descriptions.csv`
- `artifacts/numeric_features.csv`
- `artifacts/vision_errors.csv` (only created when Vision extraction issues occur)
- `artifacts/text_vectorizer.joblib`
- `artifacts/embedding_backend.json`
- `artifacts/clip_metadata.joblib` (CLIP mode only)
- `artifacts/lightgbm_ranker.joblib` (optional)

## Setup
> **Git LFS required** — images are stored in Git LFS. Install it from https://git-lfs.com before cloning, or run `git lfs pull` after cloning to download real image files instead of pointer text.

1. Install dependencies:
```bash
python -m pip install uv
uv sync --all-extras
git lfs install
git lfs pull
```
2. Create local env file:
```bash
copy .env.example .env
```
3. Set credentials in `.env`:
- `GOOGLE_APPLICATION_CREDENTIALS`
- `GOOGLE_CLOUD_PROJECT`

## Usage
### Automatic Startup Behavior
- On API and Streamlit startup, the project checks for required artifacts in `artifacts/`.
- By default, startup is inference-only and does not auto-build missing artifacts.
- If artifacts are missing, run `make build-features` explicitly.
- Optional maintainer override: set `MET_AUTO_BUILD_ON_STARTUP=true` to allow startup rebuild.
- If Vision output is missing and credentials are not available, the app shows a clear message to place the key at `config/service_account.json` and set `GOOGLE_APPLICATION_CREDENTIALS` in `.env`.

### Quickstart (Consumer)
Use this when artifacts are already present and you do not want to call Google Vision.
1. Install dependencies:
```bash
make setup
```
2. Ensure `.env` contains:
- `MET_ENABLE_VISION=false`
3. Start app:
```bash
make streamlit
```
4. Optional API service:
```bash
make serve
```

### Rebuild Pipeline (Maintainer)
Use this only when you want to regenerate recommendation artifacts.
1. Set in `.env`:
- `MET_ENABLE_VISION=true`
- `GOOGLE_APPLICATION_CREDENTIALS=<service-account-path>`
- `GOOGLE_CLOUD_PROJECT=<project-id>`
- Put your service account file at `config/service_account.json`
2. Build features:
```bash
make build-features
```
Optional TF-IDF feature-fusion controls:
```bash
uv run python -m scripts.build_features \
  --embedding-backend tfidf \
  --pca-variance 0.95 \
  --pca-max-components 256 \
  --text-weight 1.0 \
  --vision-weight 1.0 \
  --numeric-weight 1.0
```
Optional CLIP retrieval build:
```bash
uv run python -m scripts.build_features \
  --embedding-backend clip \
  --clip-model-name ViT-B-32 \
  --clip-pretrained laion2b_s34b_b79k \
  --clip-device cpu \
  --clip-batch-size 32 \
  --clip-text-weight 0.5 \
  --clip-image-weight 0.5
```
3. Optional ranker retraining:
```bash
make train-ranker
```
4. Start Streamlit/API:
```bash
make streamlit
make serve
```
Single-command backend-specific UI launchers:
```bash
make streamlit-tfidf
make streamlit-clip
```

### Quality Checks Before PR
```bash
make format
make lint
make type
make test
```

### Backend Comparison Metrics (Report Table)
After building two artifact sets (for example `artifacts_tfidf` and `artifacts_clip`), run:
```bash
uv run python -m scripts.evaluate_backends --artifacts artifacts_tfidf artifacts_clip --k 10
```
Optional JSON export:
```bash
uv run python -m scripts.evaluate_backends \
  --artifacts artifacts_tfidf artifacts_clip \
  --k 10 \
  --json-out artifacts/backend_metrics.json
```
This prints:
- `Recall@10`
- `NDCG@10`
- `Artist Coverage`
- `Department Coverage`

## Operating Modes
### Consumer Mode (no API key required)
- Use committed artifacts only.
- Keep `MET_ENABLE_VISION=false` (default in `.env.example`).
- Run app/services without rebuilding features:
  - `make serve`
  - `make streamlit`

### Maintainer Mode (API key required)
- Rebuild features using Google Vision API.
- Set:
  - `MET_ENABLE_VISION=true`
  - `GOOGLE_APPLICATION_CREDENTIALS=<path-to-service-account-json>`
- Run:
  - `make build-features`
  - optional `make train-ranker`
- Script behavior is fail-fast: if Vision is enabled and credentials are missing, build exits with a clear error.

## Commands
- `make setup`: install/update dependencies with `uv`.
- `make lint`: run `ruff` checks.
- `make format`: run `ruff format`.
- `make type`: run `ty` checks.
- `make test`: run unit tests.
- `make coverage`: run tests with coverage + HTML report.
- `make build-features`: generate/reuse cached features.
  - Supports optional flags:
    - `--embedding-backend` (`tfidf` or `clip`)
    - `--pca-variance`
    - `--pca-max-components`
    - `--text-weight`
    - `--vision-weight`
    - `--numeric-weight`
    - `--clip-model-name`
    - `--clip-pretrained`
    - `--clip-device`
    - `--clip-batch-size`
    - `--clip-text-weight`
    - `--clip-image-weight`
- `make train-ranker`: train ranker.
- `make serve`: run FastAPI at `http://localhost:8000`.
- `make streamlit`: run Streamlit at `http://localhost:8501`.
- `make streamlit-tfidf`: run Streamlit on TF-IDF artifacts at `http://localhost:8501`.
- `make streamlit-clip`: run Streamlit on CLIP artifacts at `http://localhost:8501`.

## VS Code One-Click Run
- Shared run profiles: `.vscode/launch.json`
  - `Build Features`
  - `Train Ranker`
  - `Run Streamlit App`
  - `Run API (Uvicorn)`
  - `Run API + Streamlit`
- Shared tasks: `.vscode/tasks.json`
  - restart API/Streamlit
  - build features
  - train ranker
  - run tests
- Launch profiles are pinned to project interpreter:
  - `${workspaceFolder}\\.venv\\Scripts\\python.exe`

## Troubleshooting
- `Recommender not loaded`: run `make build-features` first or verify `artifacts/` exists.
- `Vision credential errors`: set `MET_ENABLE_VISION=false` for consumer mode, or provide valid service account for maintainer mode.
- Missing dependencies: run `make setup`.

## Docker
Run API and Streamlit together:
```bash
docker compose up --build
```

## API
- `GET /health`
- `POST /recommendations/theme`
  - body: `{ "theme": "ancient egypt", "k": 10, "min_similarity": 0.2 }`

## Streamlit Features
- Themes (1 to 7)
- Pieces per exhibition (5 to 10)
- Minimum similarity threshold with error messaging
- Optional colors/styles/year-range filters
- Image display and per-theme grouping
- User guidance text for dataset coverage limitations

## Ranking Pipeline
- Stage 1 retrieval: cosine similarity over combined embeddings.
- Stage 2 reranking (if model exists): LightGBM reranker.
- Final score: calibrated to `0..1` for display consistency.
- Fallback: if ranker input shape mismatches, system falls back to cosine scores.

## Testing
Current test suite includes:
- data loader validation
- recommender behavior and coherence
- API health + theme endpoint
- feature builder helpers

Run:
```bash
make test
make coverage
```

## Security
- Never commit secrets or credential JSON files.
- Keep local credentials outside git-tracked files.

## GitFlow Workflow
- Branch model: `main`, `develop`, `feature/*`, `release/*`, `hotfix/*`.
- Work on `feature/*`, merge into `develop`.
- Release from `release/*` into `main` and back to `develop`.
- Tag releases (`vX.Y.Z`).

## Data Commit Policy
Data is currently present for bootstrap reproducibility.
After bootstrap, avoid committing new raw data updates.
