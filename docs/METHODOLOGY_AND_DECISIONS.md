# Methodology and Design Decisions

## Purpose
This document explains the current project methodology, why key design choices were made, and which legacy components remain for academic traceability.

## Problem Framing
Goal: recommend coherent themed MET exhibition items from user prompts while keeping the system reproducible, inspectable, and practical to run locally.

Constraints:
- limited project timeline
- no manual relabeling pipeline
- need for transparent artifacts and reproducible runs

## Final Operational Architecture

### Retrieval and Ranking
- Stage 1 retrieval: CLIP-based similarity over prebuilt embeddings.
- Stage 2 ranking: XGBoost learning-to-rank reranker.
- Final selection: diversity and threshold rules in recommender logic.

### Runtime Interfaces
- Streamlit UI (`make streamlit`) and FastAPI (`make serve`) are both implemented.
- They were both kept to validate the system in two usage modes (interactive UI and programmatic API).
- You do not need both running at the same time for normal use; either interface is sufficient.

### Artifacts
Core artifacts used at runtime:
- `artifacts/embeddings.npz`
- `artifacts/meta.csv`
- `artifacts/tokens.json`
- `artifacts/clip_metadata.joblib` (CLIP path)
- `artifacts/xgboost_ranker.json` (reranker)

## Evolution of the Method

### Phase 1: Vision-centric prototype (legacy)
- The project initially explored Google Vision API enrichment.
- That runtime dependency is now deprecated/removed from the active flow.
- Credential setup for Vision is not required for current operation.

### Phase 2: CLIP-first retrieval
- Retrieval moved to CLIP to better align text prompts with visual content in one embedding space.
- This reduced dependence on brittle token overlap.

### Phase 3: XGBoost reranking
- XGBoost LTR replaced LightGBM for reranking.
- Reranker uses engineered candidate features (embedding and metadata-derived signals) on top of retrieval candidates.

## Legacy Components Kept on Purpose

### TF-IDF backend
- Retained as legacy baseline context for reproducibility and academic comparison.
- Not the primary operational path in current deployment.

### Deprecated Vision references
- Some legacy structures remain in repository history/code paths to document process evolution.
- They are not required for current build/run workflow.

## Fallback and Reliability Decisions
- If `xgboost_ranker.json` is missing or reranker prediction fails, system falls back to similarity-only ranking.
- This degraded mode is explicit through warnings/diagnostics (UI/API), not silent.
- Rationale: preserve service continuity while exposing model-state transparency.

## Why We Did Not Adopt Heavier Models Yet
- More complex rerankers (for example cross-encoders) may improve quality but increase engineering and evaluation burden.
- For this academic phase, the team prioritized:
  - stable end-to-end operation
  - explainable behavior
  - reproducible artifacts and testing workflow

## Known Risks and Open Limitations
- Very high offline ranking metrics can indicate split leakage or easy validation conditions.
- Negation/query intent handling is heuristic and not perfect.
- Dataset coverage can still produce semantically nearby but curator-incorrect results.

These limitations are acknowledged and documented rather than hidden.

## Operational Workflow
1. Build artifacts: `make build-features`
2. Train reranker: `make train-ranker`
3. Run one interface:
- UI: `make streamlit`
- API: `make serve`
4. Validate quality gates: `make format && make lint && make type && make test`

## Decision Summary
- Primary backend: CLIP
- Primary reranker: XGBoost LTR
- TF-IDF: legacy baseline/documentation context
- Vision API: deprecated from active flow
- Interfaces: both available, either one sufficient in practice
