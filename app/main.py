from fastapi import FastAPI

from app.config import get_settings
from app.routers import auth, cards


def create_app() -> FastAPI:
    # A factory instead of a bare global lets tests build a fresh app
    # (with overridden settings/dependencies) without import-time side effects.
    settings = get_settings()
    app = FastAPI(title="Phocab", debug=settings.app_env == "development")

    @app.get("/health")
    def health() -> dict[str, str]:
        # Liveness only: no DB or model calls, so the load balancer and the
        # post-deploy smoke test don't fail because a dependency is slow.
        return {"status": "ok"}

    app.include_router(auth.router)
    app.include_router(cards.router)
    return app


app = create_app()
