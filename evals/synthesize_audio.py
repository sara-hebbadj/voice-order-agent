"""Turn the 30 text scripts into caller audio with OpenRouter text-to-speech.

  python -m evals.synthesize_audio --plan    # no API calls: list files, characters, cost
  python -m evals.synthesize_audio           # needs OPENROUTER_API_KEY

Writes evals/audio/<call_id>/turn_<n>.wav (git-ignored), evals/audio/manifest.csv and a
cost summary in evals/results/. The callers use a different TTS model from the agent
(MODEL_TTS_CALLER, default google/gemini-3.8-flash-lite-tts) because it follows
`instructions`: calls tagged "fast" ask it to speak quickly. Voices rotate between calls;
calls tagged "noisy" get white noise mixed in at 10 dB SNR. To use a real recording
instead (for example Sara's own voice, with consent), replace the WAV file with the
same name and set `source` to "human" in the manifest.
"""

import argparse
import csv
import io
import json
import wave
from datetime import date

import numpy as np

from evals.run import AUDIO_DIR, CALLS, EVALS, load_calls
from voice_agent.audio_client import MissingKeyError, OpenRouterAudioClient
from voice_agent.config import load_settings

# Voice names differ per TTS model (see `supported_voices` in the models API). These are
# Gemini TTS voices, a mix of female and male (Google's descriptions).
VOICES = ["Kore", "Puck", "Aoede", "Charon", "Leda", "Fenrir"]
# OpenRouter's "pcm" output is 24 kHz, 16-bit, mono (the response header said
# "audio/pcm;rate=24000;channels=1" for every model we tried on 8 October 2026).
PCM_SAMPLE_RATE = 24_000
NOISE_SNR_DB = 10

CALLER_STYLE = {
    "en": "You are a customer phoning a skincare shop about your order. Speak naturally, "
          "like a real phone call.",
    "ar": "You are a customer phoning a skincare shop about your order. Speak natural Gulf "
          "Arabic, like a real phone call.",
}


def caller_instructions(call: dict) -> str:
    text = CALLER_STYLE[call["language"]]
    if "fast" in call["tags"]:
        text += " Speak quickly, like a busy caller."
    return text


def pcm_to_wav(pcm: bytes, sample_rate: int = PCM_SAMPLE_RATE) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


def add_noise(pcm: bytes, snr_db: float = NOISE_SNR_DB, seed: int = 42) -> bytes:
    """Mix in white noise so the speech is `snr_db` decibels louder than the noise."""
    speech = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    if speech.size == 0:
        return pcm
    speech_power = np.mean(speech**2)
    noise_power = speech_power / (10 ** (snr_db / 10))
    noise = np.random.default_rng(seed).normal(0, np.sqrt(noise_power), speech.size)
    return np.clip(speech + noise, -32768, 32767).astype(np.int16).tobytes()


def plan(calls: list[dict]) -> list[dict]:
    rows = []
    for number, call in enumerate(calls):
        for index, text in enumerate(call["turns"]):
            rows.append({
                "call_id": call["call_id"], "turn": index + 1, "language": call["language"],
                "voice": VOICES[number % len(VOICES)], "tags": " ".join(call["tags"]),
                "chars": len(text), "source": "tts",
                "file": f"{call['call_id']}/turn_{index + 1}.wav", "text": text,
            })
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Synthesise the test-call audio with TTS.")
    parser.add_argument("--plan", action="store_true", help="show the plan; no API calls")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    parser.add_argument("--only", nargs="+", default=None, metavar="CALL/TURN",
                        help="re-make only these clips, e.g. en07/turn_1 (implies --force)")
    parser.add_argument("--voice", default=None,
                        help="use this voice for every clip (needed with another MODEL_TTS_CALLER)")
    args = parser.parse_args(argv)

    calls = load_calls(CALLS, args.limit)
    rows = plan(calls)
    if args.only:  # e.g. a clip whose audio did not say the scripted number
        rows = [r for r in rows if r["file"].removesuffix(".wav") in args.only]
        args.force = True
    settings = load_settings()
    chars = sum(r["chars"] for r in rows)
    print(f"{len(rows)} utterances, {chars} characters, caller TTS model "
          f"{settings.caller_tts_model} (cost is looked up after each file)")
    if args.plan:
        for r in rows[:5]:
            print(f"  {r['file']}  voice={r['voice']}  tags={r['tags'] or '-'}  {r['text']}")
        return

    try:
        client = OpenRouterAudioClient(settings, tts_model=settings.caller_tts_model,
                                       trace_path=AUDIO_DIR / "synthesis_traces.jsonl")
    except MissingKeyError as error:
        raise SystemExit(f"{error} Use --plan to preview without a key.") from error

    calls_by_id = {c["call_id"]: c for c in calls}
    generation_ids = []
    for row in rows:
        path = AUDIO_DIR / row["file"]
        if path.exists() and not args.force:
            continue
        call = calls_by_id[row["call_id"]]
        pcm = client.speak(row["text"], language=row["language"], voice=args.voice or row["voice"],
                           instructions=caller_instructions(call), response_format="pcm")
        if "noisy" in call["tags"]:
            pcm = add_noise(pcm)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pcm_to_wav(pcm))
        generation_ids.append(client.last_tts_generation_id)
        print(f"  wrote {row['file']}")

    if not args.only:
        with (AUDIO_DIR / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if generation_ids:
        costs = client.generation_costs(generation_ids)
        write_cost_summary(client.tts_model, len(generation_ids), list(costs.values()))


def write_cost_summary(model: str, files: int, costs: list[float | None]) -> None:
    """Keep the synthesis cost as evidence (the audio itself is git-ignored)."""
    known = [c for c in costs if c is not None]
    out = EVALS / "results" / f"synthesis_{date.today().isoformat()}_{files}files_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {"date": date.today().isoformat(), "caller_tts_model": model, "files": files,
               "cost_usd_known": round(sum(known), 6), "files_with_cost": len(known),
               "files_without_cost": len(costs) - len(known),
               "cost_source": "OpenRouter GET /api/v1/generation (total_cost)"}
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{files} files; cost known for {len(known)}: US${sum(known):.4f}; wrote {out}")


if __name__ == "__main__":
    main()
