"""Tests for GET /health endpoint."""

from fastapi.testclient import TestClient


def test_health_returns_200(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_body(client: TestClient) -> None:
    response = client.get("/health")
    data = response.json()
    assert data == {"status": "ok"}


def test_health_content_type(client: TestClient) -> None:
    response = client.get("/health")
    assert "application/json" in response.headers["content-type"]


# ─── /mode endpoint (Phase 13) ─────────────────────────────────────────────


def test_mode_returns_200(client: TestClient) -> None:
    """The /mode endpoint is reachable and returns the demo-mode flags."""
    response = client.get("/mode")
    assert response.status_code == 200


def test_mode_response_shape(client: TestClient) -> None:
    """The /mode response includes the four documented fields."""
    data = client.get("/mode").json()
    assert "app_env" in data
    assert "demo_mode" in data
    assert "use_demo_cv" in data
    assert "use_demo_gemini" in data


def test_mode_demo_mode_reflects_toggles() -> None:
    """`demo_mode` is true iff at least one of the demo toggles is on."""
    from app.core.config import settings
    from fastapi.testclient import TestClient
    from app.main import app
    saved_cv = settings.use_demo_cv
    saved_gem = settings.use_demo_gemini
    try:
        with TestClient(app) as c:
            settings.use_demo_cv = False
            settings.use_demo_gemini = False
            data = c.get("/mode").json()
            assert data["demo_mode"] is False

            settings.use_demo_cv = True
            data = c.get("/mode").json()
            assert data["demo_mode"] is True
            assert data["use_demo_cv"] is True

            settings.use_demo_cv = False
            settings.use_demo_gemini = True
            data = c.get("/mode").json()
            assert data["demo_mode"] is True
            assert data["use_demo_gemini"] is True
    finally:
        settings.use_demo_cv = saved_cv
        settings.use_demo_gemini = saved_gem
