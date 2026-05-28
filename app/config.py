"""Runtime configuration loaded from environment.

Only PYTHON_HMAC_SECRET is effectively required (the service fails closed
without it). Missing OpenAI / Google credentials don't crash anything — the
OCR ladder degrades gracefully and the response carries an errorMessage so the
backend keeps the document in NEEDS_REVIEW.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Auth — shared secret with the NestJS backend (HMAC-SHA256 over the body).
    python_hmac_secret: str = ""

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_vision_model: str = "gpt-4o-mini"

    # Google Vision — either the raw service-account JSON pasted into
    # GOOGLE_VISION_CREDENTIALS_JSON (easy on Railway), or rely on the SDK
    # picking up GOOGLE_APPLICATION_CREDENTIALS as a file path.
    google_vision_credentials_json: str = ""
    google_application_credentials: str = ""

    # Redis cache (falls back to in-process LRU when unset).
    redis_url: str = ""
    cache_ttl_seconds: int = 60 * 60 * 24 * 30  # 30 days
    cache_max_items: int = 512  # in-memory fallback ceiling

    # OCR cost controls
    max_vision_pages: int = 3
    vision_cost_cents_per_page: float = 0.15  # ~$1.50 / 1000 pages
    native_text_min_chars: int = 120  # below this a PDF is treated as scanned

    # Decision thresholds (parser-side hinting; the backend owns auto-approve).
    high_confidence: float = 0.90

    # Networking / limits
    http_timeout_seconds: float = 30.0
    max_file_mb: int = 25

    def vision_configured(self) -> bool:
        return bool(self.google_vision_credentials_json or self.google_application_credentials)


@lru_cache
def get_settings() -> Settings:
    return Settings()
