"""Settings for the voice agent, read from a .env file or environment variables.

Keys are never printed or logged. Real environment variables win over .env values.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

# Model IDs checked against the OpenRouter audio docs on 8 October 2026.
# Check https://openrouter.ai/models for the current list before a live run.
DEFAULT_STT_MODEL = "openai/gpt-4o-transcribe"
DEFAULT_TTS_MODEL = "openai/gpt-4o-mini-tts-2025-12-15"
DEFAULT_TTS_VOICE = "alloy"


def load_env_files() -> None:
    """Load repo/.env if present, otherwise the shared Portfolio Projects/.env."""
    candidates = [REPO_ROOT / ".env", REPO_ROOT.parents[1] / ".env"]
    for path in candidates:
        if path.is_file():
            load_dotenv(path, override=False)
            return


def _float_or_none(value: str | None) -> float | None:
    return float(value) if value else None


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    base_url: str
    stt_model: str
    tts_model: str
    tts_voice: str
    order_api_url: str | None  # None = run the mock API in-process
    tts_price_per_1m_chars: float | None  # optional, for cost estimates
    max_cost_per_run_usd: float

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)


def load_settings() -> Settings:
    load_env_files()
    env = os.environ.get
    return Settings(
        api_key=env("OPENROUTER_API_KEY") or None,
        base_url=env("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1",
        stt_model=env("MODEL_STT") or DEFAULT_STT_MODEL,
        tts_model=env("MODEL_TTS") or DEFAULT_TTS_MODEL,
        tts_voice=env("TTS_VOICE") or DEFAULT_TTS_VOICE,
        order_api_url=env("ORDER_API_URL") or None,
        tts_price_per_1m_chars=_float_or_none(env("TTS_PRICE_PER_1M_CHARS")),
        max_cost_per_run_usd=float(env("MAX_COST_PER_RUN_USD") or 3),
    )
