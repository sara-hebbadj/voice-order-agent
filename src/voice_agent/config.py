"""Settings for the voice agent, read from a .env file or environment variables.

Keys are never printed or logged. Real environment variables win over .env values.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

# Model IDs checked against OpenRouter's live model lists on 8 October 2026:
#   GET /api/v1/models?output_modalities=transcription   (speech-to-text)
#   GET /api/v1/models?output_modalities=speech          (text-to-speech)
# The OpenRouter docs still show openai/gpt-4o-mini-tts-2025-12-15 as an example, but it
# was not in the speech list that day (its /endpoints page returned 404), so it is not used.
# Speech-to-text: chosen by a 4-model comparison on the 30 test calls (README, Results).
# It follows the `language` hint (openai/gpt-4o-transcribe ignored it) and writes spoken
# numbers as numerals ("ten thousand nine hundred ninety-nine" -> "10999").
DEFAULT_STT_MODEL = "elevenlabs/scribe-v2"
# The agent's voice: the lowest TTS latency in a 3-model probe (about 0.35 s per reply),
# lists 32 languages including Arabic, priced per character.
DEFAULT_TTS_MODEL = "elevenlabs/eleven-flash-v2.5"
DEFAULT_TTS_VOICE = "sarah"  # voice names depend on the model (see `supported_voices`)
# The test callers' voices (evals/synthesize_audio.py): a different model family from the
# agent, and it follows `instructions`, so "speak quickly" and "Gulf Arabic" can be asked for.
DEFAULT_CALLER_TTS_MODEL = "google/gemini-3.8-flash-lite-tts"
# US$ per 1M characters, from the models API on 8 October 2026; used only for the budget
# guard during a run. Reported costs come from OpenRouter (usage.cost / generation API).
KNOWN_TTS_PRICE_PER_1M_CHARS = {"elevenlabs/eleven-flash-v2.5": 20.0}


def load_env_files() -> None:
    """Load repo/.env if present, otherwise the shared Portfolio Projects/.env."""
    # .parent.parent, not .parents[1]: on a Hugging Face Space the repo is /app (one parent only).
    candidates = [REPO_ROOT / ".env", REPO_ROOT.parent.parent / ".env"]
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
    caller_tts_model: str = DEFAULT_CALLER_TTS_MODEL  # test-call audio only

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)


def load_settings() -> Settings:
    load_env_files()
    env = os.environ.get
    tts_model = env("MODEL_TTS") or DEFAULT_TTS_MODEL
    tts_price = (_float_or_none(env("TTS_PRICE_PER_1M_CHARS"))
                 or KNOWN_TTS_PRICE_PER_1M_CHARS.get(tts_model))
    return Settings(
        api_key=env("OPENROUTER_API_KEY") or None,
        base_url=env("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1",
        stt_model=env("MODEL_STT") or DEFAULT_STT_MODEL,
        tts_model=tts_model,
        tts_voice=env("TTS_VOICE") or DEFAULT_TTS_VOICE,
        order_api_url=env("ORDER_API_URL") or None,
        tts_price_per_1m_chars=tts_price,
        max_cost_per_run_usd=float(env("MAX_COST_PER_RUN_USD") or 3),
        caller_tts_model=env("MODEL_TTS_CALLER") or DEFAULT_CALLER_TTS_MODEL,
    )
