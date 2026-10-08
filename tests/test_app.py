"""Smoke test for the Gradio demo (skipped when gradio is not installed, e.g. in CI)."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("gradio")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import app as demo  # noqa: E402
from voice_agent.audio_client import FakeAudioClient, fake_audio  # noqa: E402


def test_ui_builds():
    assert demo.build_ui() is not None


def test_text_turns(orders):
    session, chat, *_ = demo.new_call()
    for line in ["My order is LS-10154", "yes", orders["LS-10154"]["phone4"]]:
        session, chat, _, calls, latency = demo.on_text(line, session, chat)
    assert "shipped" in chat[-1]["content"]
    assert [c["tool"] for c in calls] == ["get_order", "get_tracking"]
    assert len(latency) == 3


def test_voice_turn_with_fake_client(tmp_path, monkeypatch):
    monkeypatch.setattr(demo, "audio_client", FakeAudioClient())
    recording = tmp_path / "mic.wav"
    recording.write_bytes(fake_audio("My order is LS-10154"))
    session, chat, *_ = demo.new_call()
    session, chat, reply_path, _, latency = demo.on_voice(str(recording), session, chat)
    assert chat[-2]["content"] == "My order is LS-10154"
    assert Path(reply_path).read_bytes() == fake_audio(chat[-1]["content"])
    assert latency[0][2] is not None  # stt time recorded
