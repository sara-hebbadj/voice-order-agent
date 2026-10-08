"""One spoken turn: caller audio -> speech-to-text -> agent -> text-to-speech.

Each stage is timed separately, because "where does the latency come from?" is the
first question to answer before trying to make a voice bot faster.
"""

import time
from dataclasses import dataclass

from voice_agent.agent import AgentTurn, OrderStatusAgent
from voice_agent.audio_client import AudioClient, Transcript


@dataclass
class VoiceTurn:
    transcript: Transcript
    agent_turn: AgentTurn
    reply_audio: bytes
    timings_ms: dict  # stt, agent, tts, total
    stt_language: str | None = None  # the language hint sent to speech-to-text, if any


def _ms(start: float, end: float) -> float:
    return round((end - start) * 1000, 1)


def stt_language_hint(agent: OrderStatusAgent) -> str | None:
    """The call's language once the agent knows it, else None (let STT detect it).

    Why: with no hint, STT guesses the language from the audio, and a one-word answer is
    too short to guess from. In the first live run "نعم" (yes) came back as "Neam." or
    "Hey," and "That's right." as German "Das Red.", so the caller's yes was lost."""
    return agent.state.language if agent.state.language_known else None


def run_voice_turn(agent: OrderStatusAgent, audio_client: AudioClient, audio: bytes,
                   filename: str = "speech.wav", voice: str | None = None,
                   response_format: str = "mp3") -> VoiceTurn:
    language = stt_language_hint(agent)
    t0 = time.perf_counter()
    transcript = audio_client.transcribe(audio, filename=filename, language=language)
    t1 = time.perf_counter()
    agent_turn = agent.handle(transcript.text)
    t2 = time.perf_counter()
    reply_audio = audio_client.speak(agent_turn.reply, language=agent_turn.language,
                                     voice=voice, response_format=response_format)
    t3 = time.perf_counter()
    timings = {"stt": _ms(t0, t1), "agent": _ms(t1, t2), "tts": _ms(t2, t3), "total": _ms(t0, t3)}
    return VoiceTurn(transcript, agent_turn, reply_audio, timings, language)


def run_text_turn(agent: OrderStatusAgent, text: str) -> tuple[AgentTurn, dict]:
    """Same as run_voice_turn but typed text in, text out (no speech, no key needed)."""
    t0 = time.perf_counter()
    agent_turn = agent.handle(text)
    agent_ms = _ms(t0, time.perf_counter())
    return agent_turn, {"stt": None, "agent": agent_ms, "tts": None, "total": agent_ms}
