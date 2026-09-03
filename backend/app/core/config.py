"""
Application configuration loaded from environment variables via pydantic-settings.
"""

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ─── Application ──────────────────────────────────────────────────────────
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # ─── Database ─────────────────────────────────────────────────────────────
    # Required in Phase 2+. Left optional here so the backend starts without a
    # running PostgreSQL instance during Phase 1 development.
    database_url: str = (
        "postgresql+psycopg2://claimsight:claimsight@localhost:5432/claimsight_db"
    )

    # ─── CORS ─────────────────────────────────────────────────────────────────
    # Stored as a raw string to remain compatible with the comma-separated
    # .env format that pydantic-settings v2 cannot JSON-decode automatically.
    # Parsed into a list of origins via the cors_origins_list property.
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    # ─── File Storage ─────────────────────────────────────────────────────────
    upload_dir: str = "../data/uploads"

    # ─── External APIs (Phase 5+) ─────────────────────────────────────────────
    gemini_api_key: str | None = None

    # ─── Demo mode (Phase 13) ─────────────────────────────────────────────────
    # When `use_demo_cv` is true the CV service returns a deterministic
    # prediction derived from the image filename rather than loading
    # the trained model. This lets the demo run end-to-end on a laptop
    # that does not have the trained CV checkpoint available. Production
    # deployments leave this False.
    use_demo_cv: bool = False
    # When `use_demo_gemini` is true the Gemini client returns a
    # canned investigation summary built from the claim's risk signals
    # rather than calling the real Gemini API. The same shape and
    # validator are used; only the response is deterministic.
    use_demo_gemini: bool = False

    # ─── Gemini client (Phase 8) ──────────────────────────────────────────────
    # The default model is the current Gemini 2.5 family. The endpoint URL
    # and timeout are exposed for tests (httpx-mock style) and for ops
    # tuning without code changes.
    gemini_model: str = "gemini-2.5-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com"
    gemini_timeout_seconds: float = 15.0
    # 2s backoff per the blueprint; exposed so the test suite can shrink it
    # in CI without changing the production behaviour.
    gemini_retry_backoff_seconds: float = 2.0


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (singleton pattern)."""
    return Settings()


settings: Settings = get_settings()
