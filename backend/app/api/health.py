"""Health check endpoint — GET /health."""

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import settings

router = APIRouter()


class HealthResponse(BaseModel):
    status: str


class ModeResponse(BaseModel):
    """Surfaces backend mode to the frontend.

    The frontend uses this to display a "demo data" indicator in the
    sidebar when the backend is running with `use_demo_cv=True` or
    `use_demo_gemini=True`. Production deployments leave both flags
    false and the indicator is hidden.
    """
    app_env: str
    demo_mode: bool
    use_demo_cv: bool
    use_demo_gemini: bool


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health() -> HealthResponse:
    """Return application health status.

    Used by load balancers, monitoring tools, and the frontend to verify
    that the backend is reachable before submitting requests.
    """
    return HealthResponse(status="ok")


@router.get("/mode", response_model=ModeResponse, summary="Backend mode")
async def mode() -> ModeResponse:
    """Return which demo-mode toggles are active.

    The frontend uses this to surface a small "Demo data" badge in
    the sidebar so reviewers can tell at a glance that the system
    is running with the deterministic stub (no real CV, no real
    Gemini). Production deployments return all flags false.
    """
    return ModeResponse(
        app_env=settings.app_env,
        demo_mode=settings.use_demo_cv or settings.use_demo_gemini,
        use_demo_cv=settings.use_demo_cv,
        use_demo_gemini=settings.use_demo_gemini,
    )
