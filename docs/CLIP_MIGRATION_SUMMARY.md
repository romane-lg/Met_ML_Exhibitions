# CLIP Migration Summary

## Why We Added CLIP
The original retrieval stack used TF-IDF/LDA features. That baseline is still supported.  
CLIP was added to improve semantic matching by embedding text and images in a shared vector space.

## What Changed
1. Added OpenCLIP-based embedding support.
2. Added backend switching (`tfidf` or `clip`) at build time.
3. Added backend metadata artifact: `artifacts/embedding_backend.json`.
4. Added CLIP metadata artifact: `artifacts/clip_metadata.joblib` in CLIP mode.
5. Updated recommender loading so it can route query encoding by backend.
6. Updated bootstrap to inference-only default (no auto-rebuild unless explicitly enabled).
7. Added `clip_tuned` backend with text-adapter checkpoint support.
8. Added CLIP adapter training script and split manifest output.
9. Added guardrail selector script to prevent portrait metric regressions.

## What Stayed the Same
1. Existing TF-IDF/LDA flow remains available.
2. Diversity constraints and reranking logic are unchanged.
3. API and Streamlit user flows are unchanged.
4. Runtime remains inference-only with prebuilt artifacts.

## How To Run
### TF-IDF Baseline
```bash
uv run python -m scripts.build_features --embedding-backend tfidf
```

### CLIP Backend
```bash
uv run python -m scripts.build_features \
  --embedding-backend clip \
  --clip-model-name ViT-B-32 \
  --clip-pretrained laion2b_s34b_b79k \
  --clip-device cpu
```

### CLIP-Tuned Backend
```bash
uv run python scripts/train_clip_lora.py --artifacts-dir artifacts_clip_tuned
uv run python -m scripts.build_features \
  --embedding-backend clip_tuned \
  --clip-tuned-checkpoint artifacts_clip_tuned/clip_text_adapter.pt
```

## Data Coverage and Prompt Guidance
The collection now targets balanced theme buckets so prompts map to available content:
- portraits/people
- landscape/nature
- religion/myth
- architecture/city
- objects/decorative arts
- abstract/patterns

Prompt guidance shown in the app:
- Use 1-3 concrete themes.
- Combine subject + style/material + period/culture.
- If results are weak, refine with material, century, culture, or style.

## Glossary
- Embedding: a numeric vector representing an item.
- Cosine similarity: a score of directional similarity between vectors.
- PCA: dimensionality reduction that preserves most variance.
- Reranking: second-stage score refinement after initial retrieval.

## Before vs After (High Level)
- Before: text/vectorizer-centric retrieval (TF-IDF/LDA).
- After: dual-mode retrieval (`tfidf` baseline + `clip` option) with the same serving interface.

## Presentation Talking Points
1. We preserved reproducibility via artifact-based serving.
2. We added CLIP for stronger multimodal semantics.
3. We kept backward compatibility for existing artifacts.
4. We enforce inference-only startup by default for reliability.
5. We added a portrait non-regression guardrail for safer CLIP tuning rollouts.

## Metrics Template
Fill these for your final report:

| Backend | Recall@10 | NDCG@10 | Artist Coverage | Department Coverage | Notes |
|---|---:|---:|---:|---:|---|
| TF-IDF |  |  |  |  |  |
| CLIP |  |  |  |  |  |

Use this command to fill the table directly:
```bash
uv run python -m scripts.evaluate_backends --artifacts artifacts_tfidf artifacts_clip --k 10
```
