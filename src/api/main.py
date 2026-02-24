from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import router
from src.bootstrap import ensure_artifacts
from src.config import get_settings
from src.models import ExhibitionRecommender


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "recommender", None) is None:
        settings = get_settings()
        status = ensure_artifacts(settings)
        app.state.bootstrap_error = status.error
        app.state.bootstrap_warning = status.warning
        if status.ready:
            try:
                app.state.recommender = ExhibitionRecommender.from_artifacts(settings.artifacts_dir)
            except FileNotFoundError:
                app.state.recommender = None
        else:
            app.state.recommender = None
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="MET Exhibition API", lifespan=lifespan)
    app.include_router(router)

    return app


app = create_app()
