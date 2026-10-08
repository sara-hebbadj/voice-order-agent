"""Transcribe chosen test-call clips with several speech-to-text models.

  python -m evals.crosscheck_stt en07/turn_1 ar13/turn_1 --models openai/whisper-large-v3 ...
  python -m evals.crosscheck_stt --all-with-digits --models <models other than MODEL_STT>

Why: when a call fails on a number, either the STT misheard it or the synthetic caller
audio never said it. If several different STT models all hear the same wrong number, the
audio is the likely cause (listen to the clip to be sure). Each clip is sent with the
call's language as the hint. Results go to evals/results/stt_crosscheck_<date>.jsonl.
"""

import argparse
import dataclasses
import hashlib
import json
from datetime import date

from evals.run import AUDIO_DIR, CALLS, EVALS, load_calls
from voice_agent.audio_client import OpenRouterAudioClient
from voice_agent.config import load_settings
from voice_agent.parser import extract_digit_groups

DEFAULT_MODELS = ["openai/whisper-large-v3", "openai/gpt-transcribe", "elevenlabs/scribe-v2",
                  "openai/gpt-4o-transcribe"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cross-check clips with several STT models.")
    parser.add_argument("clips", nargs="*", help="call_id/turn_n, e.g. en07/turn_1")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--all-with-digits", action="store_true",
                        help="every clip whose script has a number (to check the test audio)")
    args = parser.parse_args(argv)

    calls = {c["call_id"]: c for c in load_calls(CALLS, None)}
    if args.all_with_digits:
        args.clips = [f"{c['call_id']}/turn_{i + 1}" for c in calls.values()
                      for i, line in enumerate(c["turns"]) if extract_digit_groups(line)]
    flagged = []
    settings = load_settings()
    out = EVALS / "results" / f"stt_crosscheck_{date.today().isoformat()}.jsonl"
    with out.open("a", encoding="utf-8") as f:
        for clip in args.clips:
            call_id, turn = clip.split("/")
            call = calls[call_id]
            script = call["turns"][int(turn.removeprefix("turn_")) - 1]
            audio = (AUDIO_DIR / f"{clip}.wav").read_bytes()
            record = {"clip": clip, "audio_sha256": hashlib.sha256(audio).hexdigest()[:12],
                      "language": call["language"], "script": script,
                      "script_digits": extract_digit_groups(script), "heard": {}}
            for model in args.models:
                client = OpenRouterAudioClient(dataclasses.replace(settings, stt_model=model))
                transcript = client.transcribe(audio, filename="clip.wav",
                                               language=call["language"])
                record["heard"][model] = {"text": transcript.text,
                                          "digits": extract_digit_groups(transcript.text),
                                          "cost_usd": transcript.usage.get("cost")}
            # Flag for listening when most models hear different digits from the script.
            # Not proof of bad audio: "2790" said as "twenty-seven ninety" is flagged too.
            wrong = sum(h["digits"] != record["script_digits"] for h in record["heard"].values())
            record["flag"] = wrong > len(args.models) / 2
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if record["flag"]:
                flagged.append(clip)
            print(clip, repr(script), {m: h["text"] for m, h in record["heard"].items()})
    print(f"{len(args.clips)} clips, flagged {len(flagged)}: {' '.join(flagged)}\nwrote {out}")


if __name__ == "__main__":
    main()
