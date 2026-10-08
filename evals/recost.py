"""Fill in text-to-speech costs that were missing when a live run finished.

  python -m evals.recost evals/results/<run>_turns.jsonl

OpenRouter's generation records can appear minutes late, or not at all. This asks again
for the missing ones, falls back to characters x per-character price (exact for the
default ElevenLabs voice), and rewrites the cost fields of the run's turns, calls and
summary files. Nothing else in the files changes.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

from evals.run import fill_missing_tts_costs, total_calls
from voice_agent.audio_client import OpenRouterAudioClient
from voice_agent.config import KNOWN_TTS_PRICE_PER_1M_CHARS, load_settings


def recost(turns_path: Path) -> dict:
    stem = turns_path.name.removesuffix("_turns.jsonl")
    calls_path = turns_path.with_name(f"{stem}_calls.csv")
    summary_path = turns_path.with_name(f"{stem}_summary.json")
    turns = [json.loads(line) for line in turns_path.read_text(encoding="utf-8").splitlines()]
    with calls_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    for turn in turns:  # runs saved before tts_cost_source existed
        turn.setdefault("tts_cost_source",
                        "generation_api" if turn["tts_cost_usd"] is not None else None)
    missing = [t["tts_generation_id"] for t in turns if t["tts_cost_usd"] is None]
    found = OpenRouterAudioClient(load_settings()).generation_costs(missing, passes=1)
    for turn in turns:
        if turn["tts_cost_usd"] is None and found.get(turn["tts_generation_id"]) is not None:
            turn["tts_cost_usd"] = found[turn["tts_generation_id"]]
            turn["tts_cost_source"] = "generation_api"
    counts = fill_missing_tts_costs(turns, KNOWN_TTS_PRICE_PER_1M_CHARS.get(summary["tts_model"]))
    total_calls(rows, turns)

    for lang in ("all", "en", "ar"):
        costs = [float(r["cost_usd"]) for r in rows if lang == "all" or r["language"] == lang]
        mean = round(float(np.mean(costs)), 5)
        if lang == "all":
            summary["cost_per_call_usd"] = mean
        else:
            summary[lang]["cost_per_call_usd"] = mean
    summary["total_cost_usd"] = round(sum(float(r["cost_usd"]) for r in rows), 6)
    summary.pop("tts_replies_without_cost", None)
    summary["tts_cost_sources"] = counts
    summary["cost_note"] = "TTS costs filled in after the run by evals/recost.py"

    turns_path.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in turns),
                          encoding="utf-8")
    with calls_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    return counts


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(arg, recost(Path(arg)))
