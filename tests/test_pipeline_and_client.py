import json

import pytest

from voice_agent.audio_client import (
    FakeAudioClient,
    MissingKeyError,
    OpenRouterAudioClient,
    fake_audio,
)
from voice_agent.config import Settings
from voice_agent.pipeline import run_voice_turn

SETTINGS = Settings(
    api_key="sk-test-not-a-real-key", base_url="https://openrouter.ai/api/v1",
    stt_model="openai/gpt-4o-transcribe", tts_model="openai/gpt-4o-mini-tts-2025-12-15",
    tts_voice="alloy", order_api_url=None, tts_price_per_1m_chars=None,
    max_cost_per_run_usd=3,
)


def test_voice_turn_with_fake_client_times_every_stage(agent):
    turn = run_voice_turn(agent, FakeAudioClient(), fake_audio("My order is LS-10154"))
    assert turn.transcript.text == "My order is LS-10154"
    assert turn.reply_audio == fake_audio(turn.agent_turn.reply)
    assert set(turn.timings_ms) == {"stt", "agent", "tts", "total"}
    assert turn.timings_ms["total"] >= turn.timings_ms["agent"]


def test_real_client_needs_a_key():
    from dataclasses import replace

    with pytest.raises(MissingKeyError):
        OpenRouterAudioClient(replace(SETTINGS, api_key=None))


def test_real_client_request_shape_without_network(tmp_path):
    """Check endpoints and payloads with a mock HTTP transport (no network)."""
    httpx2 = pytest.importorskip("httpx2")  # the HTTP library used by openai>=3
    seen = []

    def handler(request):
        seen.append(request)
        if request.url.path.endswith("/audio/transcriptions"):
            return httpx2.Response(200, json={"text": "10045", "usage": {"cost": 0.0001}})
        return httpx2.Response(200, content=b"mp3-bytes", headers={"x-generation-id": "g1"})

    trace = tmp_path / "traces.jsonl"
    client = OpenRouterAudioClient(
        SETTINGS, trace_path=trace,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )
    assert client.transcribe(b"RIFF", "a.wav").text == "10045"
    assert client.speak("Hello", language="ar") == b"mp3-bytes"

    stt, tts = seen
    assert stt.url.path == "/api/v1/audio/transcriptions"
    assert stt.headers["content-type"].startswith("multipart/form-data")
    assert tts.url.path == "/api/v1/audio/speech"
    body = json.loads(tts.content)
    assert body["model"] == SETTINGS.tts_model and "Arabic" in body["instructions"]

    records = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [r["kind"] for r in records] == ["stt", "tts"]
    assert records[0]["usage"]["cost"] == 0.0001
    assert SETTINGS.api_key not in trace.read_text()  # the key is never logged


class RecordingFakeClient(FakeAudioClient):
    """Remembers the language hint of every transcription request."""

    def __init__(self):
        self.languages = []

    def transcribe(self, audio, filename="speech.wav", language=None):
        self.languages.append(language)
        return super().transcribe(audio, filename, language)


def test_stt_gets_the_call_language_once_known(agent):
    # Live run 1: with no hint, a lone "نعم" was transcribed as "Neam." and the yes was lost.
    client = RecordingFakeClient()
    for line in ["رقم الطلب 10012", "نعم"]:
        run_voice_turn(agent, client, fake_audio(line))
    assert client.languages == [None, "ar"]  # first turn: let STT detect the language


def test_no_hint_while_the_language_is_unknown(agent):
    client = RecordingFakeClient()
    run_voice_turn(agent, client, fake_audio("10012"))  # digits only: no language yet
    run_voice_turn(agent, client, fake_audio("yes"))
    assert client.languages == [None, None]


def test_generation_costs_retries_until_the_record_appears():
    httpx2 = pytest.importorskip("httpx2")
    asked = []

    def handler(request):
        asked.append(request.url.params["id"])
        if request.url.path.endswith("/generation") and len(asked) > 1:
            return httpx2.Response(200, json={"data": {"total_cost": 0.0012}})
        return httpx2.Response(404, json={"error": {"message": "Not Found", "code": 404}})

    client = OpenRouterAudioClient(
        SETTINGS, http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
    client._client = client._client.with_options(max_retries=0)
    assert client.generation_costs(["gen-1"], passes=2, wait_s=0) == {"gen-1": 0.0012}
    assert asked == ["gen-1", "gen-1"]
