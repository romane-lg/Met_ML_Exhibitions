from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter()


class ThemeRequest(BaseModel):
    theme: str = Field(..., min_length=1)
    k: int = Field(10, ge=1, le=50)
    min_similarity: float = Field(0.1, ge=0.0, le=1.0)


def _reranker_status_payload(recommender: object | None) -> dict[str, object]:
    status = getattr(recommender, "last_reranker_status", None) if recommender is not None else None
    if isinstance(status, dict):
        reranker_used = bool(status.get("reranker_used", False))
        fallback_reason = status.get("fallback_reason")
        return {
            "reranker_used": reranker_used,
            "fallback_reason": fallback_reason,
        }
    return {"reranker_used": False, "fallback_reason": "status_unavailable"}


@router.get("/health")
def health(request: Request) -> dict[str, object]:
    error = getattr(request.app.state, "bootstrap_error", None)
    warning = getattr(request.app.state, "bootstrap_warning", None)
    recommender = getattr(request.app.state, "recommender", None)
    ranker_loaded = bool(getattr(recommender, "ranker", None)) if recommender is not None else False
    if error:
        return {
            "status": "degraded",
            "error": error,
            "warning": warning,
            "xgboost_reranker_loaded": ranker_loaded,
        }
    return {
        "status": "ok",
        "error": None,
        "warning": warning,
        "xgboost_reranker_loaded": ranker_loaded,
    }


@router.post("/recommendations/theme")
def recommend_for_theme(request: Request, payload: ThemeRequest):
    error = getattr(request.app.state, "bootstrap_error", None)
    warning = getattr(request.app.state, "bootstrap_warning", None)
    if error:
        raise HTTPException(status_code=503, detail=error)

    recommender = getattr(request.app.state, "recommender", None)
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not loaded")

    frame = recommender.recommend_for_theme(
        payload.theme,
        n_recommendations=payload.k,
        min_score=payload.min_similarity,
    )
    safe = frame.replace([np.inf, -np.inf], np.nan).where(pd.notna(frame), None)
    diagnostics = _reranker_status_payload(recommender)
    diagnostics["bootstrap_warning"] = warning
    return {
        "results": safe.to_dict(orient="records"),
        "diagnostics": diagnostics,
    }
