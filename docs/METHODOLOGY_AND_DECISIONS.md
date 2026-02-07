# Methodology and Design Decisions

## Purpose
This document explains the implementation additions made to productionize the project without changing the core objective (themed artwork recommendation).

## Major Additions
- Dependency and task standardization via `uv` and `Makefile`.
- Code quality gates via `ruff` (lint/format) and `ty` (type checks).
- API service with FastAPI endpoints.
- Containerized execution for API + Streamlit.
- Artifact-based runtime to avoid repeated external API calls.
- Consumer vs Maintainer operating modes.

## Tooling Stack
- Dependency manager: `uv`
- Lint/format: `ruff`
- Type checker: `ty`
- Tests: `pytest`, `pytest-cov`
- Load tests: `locust`
- Runtime: `streamlit`, `fastapi`, `uvicorn`

## Makefile Commands
- `make setup`: install dependencies with `uv`.
- `make lint`: static lint checks.
- `make format`: formatter pass.
- `make type`: type checks.
- `make test`: unit tests.
- `make coverage`: test coverage report.
- `make build-features`: generate artifacts.
- `make train-ranker`: train LightGBM ranker.
- `make serve`: run FastAPI.
- `make streamlit`: run Streamlit UI.

## Why This Makefile Pattern
- Reproducibility: same command runs locally and in CI.
- Team consistency: no personal command variants.
- Auditability: command behavior is explicit in one file.
- Lower onboarding cost for new contributors.

## Runtime Architecture
### Build Phase
- Input: `data/raw/met_data.csv`, `data/raw/images/`
- Script: `scripts/build_features.py`
- Output artifacts:
  - `artifacts/embeddings.npz`
  - `artifacts/meta.csv`
  - `artifacts/tokens.json`
  - `artifacts/descriptions.csv`
  - `artifacts/numeric_features.csv`
  - `artifacts/text_vectorizer.joblib`

### Startup Bootstrap
- API and Streamlit both run a startup bootstrap check.
- Required artifacts are validated (`embeddings.npz`, `meta.csv`, `tokens.json`, `text_vectorizer.joblib`).
- If missing, the app attempts to build them automatically.
- If Google Vision output is missing and credentials are not configured, startup surfaces a clear message to add `config/service_account.json` and set `.env` values.

### Ranker Phase (optional)
- Script: `scripts/train_ranker.py`
- Output: `artifacts/lightgbm_ranker.joblib`

## Feature/Model Improvements Implemented
### 1) Image Pipeline Split and Cleaning
- Added `src/loaders/image_api_loader.py` to isolate raw Google Vision API request/response and retry logic.
- Refactored `src/features/image_features.py` to focus on transformation only:
  - response normalization
  - stopword removal and lemmatization
  - token extraction
- Added `scripts/cleanup_invalid_images.py` to detect/remove corrupt/non-image files by binary signature.

### 2) Shared NLP Preprocessing
- Added `src/features/nlp_utils.py` for shared tokenization across image and text paths.
- Standardized stopwords + lemmatization between:
  - `src/features/text_features.py`
  - `src/features/image_features.py`
- Numeric tokens are excluded from semantic tokens by default to reduce token noise.

### 3) Numeric Signal Retention
- Added structured numeric extraction in addition to text tokens:
  - color statistics (`vision_avg_red/green/blue`)
  - vision confidence aggregates
  - OCR number stats
  - metadata year stats
- Persisted numeric output in `artifacts/numeric_features.csv` for downstream ranking.

### 4) Ranking Quality Upgrade
- Kept fast retrieval by cosine similarity, then added stage-2 reranking with LightGBM.
- Updated ranker training features to include:
  - embedding differences
  - cosine feature
  - numeric feature differences
- Added calibrated `0..1` final score output for API/UI consistency.
- Added safe fallback to base cosine scoring if ranker feature-shape mismatch occurs.

### Serve Phase
- API: `src/api/main.py`
- UI: `src/app/streamlit_app.py`
- Recommender core: `src/models/recommender.py`

## Technique Choices and Rationale
### Analytics Decision
The project uses a hybrid of image-derived and text-derived signals because themed exhibition search requires semantic coverage from both visual appearance and metadata context.

### Why TF-IDF for Text Baseline
- Deterministic and auditable feature space.
- Fast on medium-sized museum datasets.
- Works well with sparse metadata fields and short descriptions.
- Easy to inspect for debugging query mismatch.

### Why Vision-Derived Tokens for Image Signals
- Uses labels, localized objects, web entities, OCR text, and color signatures from Vision.
- Produces interpretable tokens that can be merged with text features.
- Expands semantic coverage beyond metadata-only descriptions.
- Suitable for cold-start when no interaction data exists.

### Why Combined Embeddings
- Single vector space simplifies retrieval and scoring.
- Preserves compatibility with nearest-neighbor search and ranker features.
- Supports both API and Streamlit from the same artifact set.

### Why Cosine Similarity
- Robust to magnitude differences in sparse/high-dimensional vectors.
- Standard choice for token-based embeddings.
- Efficient with pre-normalized vectors and neighbor indexing.

### Why Optional LightGBM Re-ranker
- Adds non-linear refinement over raw similarity.
- Uses pairwise-difference features without changing base retrieval pipeline.
- Kept optional to avoid forcing heavier training/runtime dependencies for all users.

### Why Theme + Filter Controls (colors/styles/years)
- Curators often think in constrained concepts, not only free-text themes.
- Post-retrieval filtering improves practical relevance while preserving recall.
- Minimal overhead compared to retraining specialized models.

### Why Minimum Similarity Threshold
- Prevents low-confidence recommendations from appearing as valid matches.
- Makes failure mode explicit with user-facing messaging.
- Improves trust in outputs for curation workflows.

### Why Artifact Caching
- Avoids repeated external API cost/latency.
- Enables consumer mode with no credentials.
- Stabilizes outputs across teammates and environments.

## Consumer vs Maintainer Modes
### Consumer Mode
- Goal: run app using prebuilt artifacts.
- Required:
  - `MET_ENABLE_VISION=false`
- No external Vision API usage.

### Maintainer Mode
- Goal: rebuild/update artifacts.
- Required:
  - `MET_ENABLE_VISION=true`
  - valid `GOOGLE_APPLICATION_CREDENTIALS`
  - valid `GOOGLE_CLOUD_PROJECT`
- If Vision is enabled and credentials are missing, build fails fast with clear errors.

## Configuration
Environment variables expected in `.env`:
- `GOOGLE_APPLICATION_CREDENTIALS`
- `GOOGLE_CLOUD_PROJECT`
- `MET_DATA_CSV`
- `MET_IMAGES_DIR`
- `MET_ARTIFACTS_DIR`
- `MET_ENABLE_VISION`

Template reference: `.env.example`

## API Surface
- `GET /health`
- `POST /recommendations/theme`
  - payload: `{"theme": "ancient egypt", "k": 10, "min_similarity": 0.2}`

## Streamlit Behavior
- Supports 1-7 themes.
- Supports 5-10 pieces per exhibit.
- Supports minimum similarity threshold.
- Supports color/style/year filtering.
- Displays images directly from `data/raw/images` using `image_path` links.

## Testing Strategy
- `tests/test_data_loader.py`: data validation and summary tests.
- `tests/test_recommender.py`: scoring, grouping, coherence behavior.
- `tests/test_api.py`: API health and recommendation endpoint.
- `tests/test_feature_builder.py`: feature builder utility checks.
- `tests/test_image_api_loader.py`: Vision loader mapping/error path tests.
- `tests/test_image_features.py`: response cleaning, tokenization, numeric extraction.
- `tests/test_nlp_utils.py`: shared tokenizer behavior.
- `locustfile.py`: baseline load test scenario.

## Containerization
### Files
- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`

### Usage
- `docker compose up --build`
- Services:
  - API on `:8000`
  - Streamlit on `:8501`

## Security and Git Hygiene
- Do not commit `config/service_account.json`.
- Keep `.env` out of git.
- Bootstrap artifacts can be committed once if needed for team startup.
- After bootstrap, avoid raw data and artifact churn in git.

## Suggested GitFlow Sequence
1. `main` stable branch.
2. `develop` integration branch.
3. `feature/*` branches per domain:
   - vision
   - nlp
   - merge-image-nlp
   - recommender
   - streamlit
   - docker
4. `release/*` for production cut.
5. Tag release (`v0.1.0`).

## Operational Notes
- Run `make setup` before running project commands.
- Run `make lint type test` before opening PRs.
- Rebuild artifacts only when model/data logic changes.
