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


@router.get("/health")
def health(request: Request) -> dict[str, str | None]:
    error = getattr(request.app.state, "bootstrap_error", None)
    warning = getattr(request.app.state, "bootstrap_warning", None)
    if error:
        return {"status": "degraded", "error": error, "warning": warning}
    return {"status": "ok", "error": None, "warning": warning}


@router.post("/recommendations/theme")
def recommend_for_theme(request: Request, payload: ThemeRequest):
    error = getattr(request.app.state, "bootstrap_error", None)
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
    return {"results": safe.to_dict(orient="records")}
