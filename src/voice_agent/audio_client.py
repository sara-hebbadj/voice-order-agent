"""Speech-to-text and text-to-speech behind one small interface.

- OpenRouterAudioClient calls OpenRouter's OpenAI-compatible audio endpoints:
  POST /api/v1/audio/transcriptions (speech-to-text) and POST /api/v1/audio/speech (TTS).
- FakeAudioClient is used by the tests and by `--dry-run`: its "audio" is just text
  with a prefix, so the whole pipeline runs with no network and no key.

Every real call is appended to a JSONL trace (model, latency, usage/cost, outcome).
The API key is never written to the trace.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from openai import NotFoundError, OpenAI

from voice_agent.config import Settings

FAKE_PREFIX = b"FAKE-AUDIO:"

# Delivery instructions for the TTS model (OpenRouter forwards them to OpenAI and Gemini TTS
# models; other providers, such as ElevenLabs, ignore them).
AGENT_VOICE_INSTRUCTIONS = {
    "en": "You are a calm, warm customer-service agent on a phone line. Speak clearly and "
          "read digits slowly, one at a time.",
    "ar": "You are a calm, warm customer-service agent on a phone line. Speak Arabic clearly "
          "with a neutral Gulf-friendly accent and read digits slowly, one at a time.",
}


@dataclass
class Transcript:
    text: str
    model: str
    usage: dict = field(default_factory=dict)  # OpenRouter returns seconds and cost here


class AudioClient(Protocol):
    def transcribe(self, audio: bytes, filename: str = "speech.wav",
                   language: str | None = None) -> Transcript: ...

    def speak(self, text: str, language: str = "en", voice: str | None = None,
              instructions: str | None = None, response_format: str = "mp3") -> bytes: ...


class MissingKeyError(RuntimeError):
    pass


def fake_audio(text: str) -> bytes:
    """Bytes that FakeAudioClient.transcribe turns back into `text`."""
    return FAKE_PREFIX + text.encode("utf-8")


class FakeAudioClient:
    """Offline stand-in for the speech APIs (tests and --dry-run only)."""

    stt_model = "fake-stt"
    tts_model = "fake-tts"
    last_tts_generation_id = None

    def generation_costs(self, generation_ids: list[str], **_) -> dict[str, float | None]:
        return {gid: 0.0 for gid in generation_ids if gid}

    def transcribe(self, audio: bytes, filename: str = "speech.wav",
                   language: str | None = None) -> Transcript:
        text = audio[len(FAKE_PREFIX):].decode("utf-8") if audio.startswith(FAKE_PREFIX) else ""
        return Transcript(text=text, model=self.stt_model, usage={"cost": 0.0})

    def speak(self, text: str, language: str = "en", voice: str | None = None,
              instructions: str | None = None, response_format: str = "mp3") -> bytes:
        return fake_audio(text)


class OpenRouterAudioClient:
    def __init__(self, settings: Settings, trace_path: Path | None = None, http_client=None,
                 tts_model: str | None = None):
        if not settings.has_key:
            raise MissingKeyError("OPENROUTER_API_KEY is not set (see .env.example).")
        self.settings = settings
        self.stt_model = settings.stt_model
        self.tts_model = tts_model or settings.tts_model
        self.trace_path = trace_path
        # The speech endpoint returns audio bytes and no usage, so we keep its generation ID
        # and ask OpenRouter for the exact cost later (see generation_cost).
        self.last_tts_generation_id: str | None = None
        self._client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=60,
            max_retries=1,
            http_client=http_client,
        )

    def transcribe(self, audio: bytes, filename: str = "speech.wav",
                   language: str | None = None) -> Transcript:
        options = {"language": language} if language else {}
        start = time.perf_counter()
        try:
            raw = self._client.audio.transcriptions.with_raw_response.create(
                model=self.stt_model, file=(filename, audio), **options
            )
            body = raw.parse().model_dump()
        except Exception as error:
            self._trace("stt", self.stt_model, start, ok=False, error=type(error).__name__)
            raise
        usage = {k: v for k, v in (body.get("usage") or {}).items() if v is not None}
        self._trace("stt", self.stt_model, start, ok=True, usage=usage,
                    generation_id=raw.headers.get("x-generation-id"),
                    audio_bytes=len(audio), text_chars=len(body.get("text") or ""))
        return Transcript(text=body.get("text") or "", model=self.stt_model, usage=usage)

    def speak(self, text: str, language: str = "en", voice: str | None = None,
              instructions: str | None = None, response_format: str = "mp3") -> bytes:
        instructions = instructions or AGENT_VOICE_INSTRUCTIONS.get(language)
        options = {"instructions": instructions} if instructions else {}
        self.last_tts_generation_id = None
        start = time.perf_counter()
        try:
            raw = self._client.audio.speech.with_raw_response.create(
                model=self.tts_model,
                voice=voice or self.settings.tts_voice,
                input=text,
                response_format=response_format,
                **options,
            )
            audio = raw.content
        except Exception as error:
            self._trace("tts", self.tts_model, start, ok=False, error=type(error).__name__)
            raise
        self.last_tts_generation_id = raw.headers.get("x-generation-id")
        self._trace("tts", self.tts_model, start, ok=True, usage=self._tts_usage(text),
                    generation_id=self.last_tts_generation_id,
                    text_chars=len(text), audio_bytes=len(audio))
        return audio

    def _tts_usage(self, text: str) -> dict:
        """OpenRouter's TTS response has no usage body, so this is only an estimate from the
        character count (TTS_PRICE_PER_1M_CHARS). The exact cost comes from generation_cost."""
        price = self.settings.tts_price_per_1m_chars
        return {"chars": len(text), "cost_estimate": len(text) * price / 1e6 if price else None}

    def generation_costs(self, generation_ids: list[str], passes: int = 3,
                         wait_s: float = 15.0) -> dict[str, float | None]:
        """Exact cost in US$ of earlier requests, from GET /api/v1/generation?id=...

        A record can take from seconds to minutes to appear (404 until then), so we ask
        for all IDs, wait, and ask again for the missing ones. IDs still missing after the
        last pass map to None, and the caller reports how many costs are unknown."""
        costs: dict[str, float | None] = {gid: None for gid in generation_ids if gid}
        for attempt in range(passes):
            if attempt:
                time.sleep(wait_s)
            for gid in [g for g, cost in costs.items() if cost is None]:
                try:
                    body = self._client.get("/generation", cast_to=object,
                                            options={"params": {"id": gid}})
                    costs[gid] = (body.get("data") or {}).get("total_cost")
                except NotFoundError:
                    pass
            if all(cost is not None for cost in costs.values()):
                break
        return costs

    def _trace(self, kind: str, model: str, start: float, **fields) -> None:
        if self.trace_path is None:
            return
        record = {
            "time": datetime.now(UTC).isoformat(timespec="seconds"),
            "kind": kind,
            "model": model,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            **fields,
        }
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
