"""Run the 30 scripted test calls through the agent and score them.

  python -m evals.run --mode text              # now: scripts as text, no speech (real numbers)
  python -m evals.run --mode audio --dry-run   # fake speech client, proves the pipeline
  python -m evals.synthesize_audio             # once a key exists: make the test-call audio
  python -m evals.run --mode audio             # live: STT -> agent -> TTS per turn

Outputs go to evals/results/ (real runs) or evals/dry_run/ (never real results):
  <run>_calls.csv, <run>_turns.jsonl, <run>_summary.json, and traces.jsonl for audio runs.
Live audio runs also save the agent's spoken replies as evals/audio/<call_id>/reply_<n>.mp3
(git-ignored) so you can listen to every call.

Cost: speech-to-text cost comes back in each response (usage.cost). Text-to-speech returns
only audio, so after the run we ask OpenRouter for each reply's exact cost by its
generation ID. During the run, the budget guard uses an estimate from the character count.
"""

import argparse
import csv
import dataclasses
import json
import sys
from datetime import date
from pathlib import Path

from evals.scoring import score_call, score_turn, summarise
from voice_agent.agent import OrderStatusAgent
from voice_agent.audio_client import (
    FakeAudioClient,
    MissingKeyError,
    OpenRouterAudioClient,
    fake_audio,
)
from voice_agent.config import load_settings
from voice_agent.pipeline import run_text_turn, run_voice_turn
from voice_agent.tools import make_order_tools

EVALS = Path(__file__).resolve().parent
CALLS = EVALS / "calls.jsonl"
AUDIO_DIR = EVALS / "audio"
DRY_RUN_LABEL = "DRY RUN with a fake speech client: NOT real results"
TEXT_ONLY_LABEL = "text-only, no speech: scripts typed into the agent (no STT/TTS)"


def load_calls(path: Path, limit: int | None) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        calls = [json.loads(line) for line in f if line.strip()]
    return calls[:limit] if limit else calls


def audio_for_turn(call: dict, index: int, dry_run: bool) -> tuple[bytes, str]:
    """Dry run: fake audio that 'transcribes' to the script. Live: the synthesised WAV."""
    if dry_run:
        return fake_audio(call["turns"][index]), "fake.wav"
    path = AUDIO_DIR / call["call_id"] / f"turn_{index + 1}.wav"
    if not path.is_file():
        sys.exit(f"Missing {path}. Run `python -m evals.synthesize_audio` first.")
    return path.read_bytes(), path.name


def run_call(call: dict, mode: str, tools, audio_client, dry_run: bool) -> tuple[dict, list]:
    agent = OrderStatusAgent(tools)
    turns, cost = [], 0.0
    for index, script in enumerate(call["turns"]):
        record = {"call_id": call["call_id"], "language": call["language"], "turn": index + 1,
                  "script": script}
        if mode == "text":
            agent_turn, timings = run_text_turn(agent, script)
            heard = script
        else:
            audio, filename = audio_for_turn(call, index, dry_run)
            voice_turn = run_voice_turn(agent, audio_client, audio, filename=filename)
            agent_turn, timings = voice_turn.agent_turn, voice_turn.timings_ms
            heard = voice_turn.transcript.text
            stt_cost = voice_turn.transcript.usage.get("cost") or 0.0
            cost += stt_cost + _tts_estimate(audio_client, agent_turn.reply)
            record |= {"stt_language_hint": voice_turn.stt_language,
                       "stt_scores": score_turn(script, heard), "stt_cost_usd": stt_cost,
                       "tts_chars": len(agent_turn.reply),
                       "tts_generation_id": audio_client.last_tts_generation_id}
            if not dry_run:
                save_reply_audio(call, index, voice_turn.reply_audio)
        turns.append(record | {
            "heard": heard, "reply": agent_turn.reply, "stage": agent_turn.stage,
            "tool_calls": [dataclasses.asdict(c) | {"result": None} for c in agent_turn.tool_calls],
            "timings_ms": timings,
        })
        if agent_turn.ended:
            break

    state = agent.state
    outcome = {"reported": state.reported, "handover_reason": state.handover_reason,
               "confirmed_order": state.confirmed_order, "language": state.language}
    row = {
        "call_id": call["call_id"], "language": call["language"], "category": call["category"],
        "expected_outcome": call["expected_outcome"], "expected_status": call["expected_status"],
        "expected_order_id": call["expected_order_id"],
        "expected_handover_reason": call["expected_handover_reason"],
        "final_stage": state.stage, "handover_reason": state.handover_reason,
        "confirmed_order": state.confirmed_order,
        "reported_status": state.reported[-1][1] if state.reported else None,
        "failures_at_end": state.failures, "turns_used": len(turns),
        "cost_usd": round(cost, 6) if mode == "audio" and not dry_run else None,
        **score_call(call, outcome),
    }
    return row, turns


def _tts_estimate(audio_client, text: str) -> float:
    """Character-count estimate, only for the budget guard while the run is going."""
    price = getattr(getattr(audio_client, "settings", None), "tts_price_per_1m_chars", None)
    return len(text) * price / 1e6 if price else 0.0


def save_reply_audio(call: dict, index: int, audio: bytes) -> None:
    path = AUDIO_DIR / call["call_id"] / f"reply_{index + 1}.mp3"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audio)


def add_exact_costs(rows: list[dict], turns: list[dict], audio_client) -> dict:
    """Replace the TTS estimates with OpenRouter's cost per reply, then total each call."""
    costs = audio_client.generation_costs([t["tts_generation_id"] for t in turns])
    for turn in turns:
        turn["tts_cost_usd"] = costs.get(turn["tts_generation_id"])
        turn["tts_cost_source"] = "generation_api" if turn["tts_cost_usd"] is not None else None
    price = getattr(audio_client.settings, "tts_price_per_1m_chars", None)
    counts = fill_missing_tts_costs(turns, price)
    total_calls(rows, turns)
    return counts


def fill_missing_tts_costs(turns: list[dict], price_per_1m_chars: float | None) -> dict:
    """Some generation records never appear in OpenRouter's API. For a model priced per
    character (the default ElevenLabs voice), characters x price is the exact cost: it
    matched every record we did get on 8 October 2026. Each turn says where its cost
    came from; turns with no cost at all stay None and are counted."""
    for turn in turns:
        if turn["tts_cost_usd"] is None and price_per_1m_chars:
            turn["tts_cost_usd"] = round(turn["tts_chars"] * price_per_1m_chars / 1e6, 8)
            turn["tts_cost_source"] = "per_char_price"
    sources = [t["tts_cost_source"] for t in turns]
    return {"generation_api": sources.count("generation_api"),
            "per_char_price": sources.count("per_char_price"), "unknown": sources.count(None)}


def total_calls(rows: list[dict], turns: list[dict]) -> None:
    for row in rows:
        call_turns = [t for t in turns if t["call_id"] == row["call_id"]]
        row["cost_usd"] = round(sum(t["stt_cost_usd"] + (t["tts_cost_usd"] or 0.0)
                                    for t in call_turns), 6)


def write_outputs(out_dir: Path, run_id: str, rows: list, turns: list, summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{run_id}_calls.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / f"{run_id}_turns.jsonl").open("w", encoding="utf-8") as f:
        for turn in turns:
            f.write(json.dumps(turn, ensure_ascii=False) + "\n")
    (out_dir / f"{run_id}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def print_summary(summary: dict) -> None:
    print(f"\n{summary['label']}\nRun: {summary['run_id']}  calls: {summary['calls']}")
    for lang in ("all", "en", "ar"):
        s = summary[lang]
        print(f"  [{lang}] task success {s['task_success']['k']}/{s['task_success']['n']}"
              f" | order capture {s['order_capture']['k']}/{s['order_capture']['n']}"
              f" | handover decisions {s['handover_decision']['k']}/{s['handover_decision']['n']}"
              f" | status leaks {s['status_leaks']} | language {s['language_ok']['k']}/"
              f"{s['language_ok']['n']}")
        if "stt" in s:
            stt = s["stt"]
            print(f"        STT: WER {stt['wer']['pct']}% ({stt['wer']['edits']}/"
                  f"{stt['wer']['ref_words']} words) | numbers {stt['number_capture']['k']}/"
                  f"{stt['number_capture']['n']} | order IDs {stt['order_id_capture']['k']}/"
                  f"{stt['order_id_capture']['n']} turns | cost/call US$"
                  f"{s.get('cost_per_call_usd', summary['cost_per_call_usd'])}")
        latencies = s.get("latency_per_turn") or (summary["latency_per_turn"]
                                                  if lang == "all" else {})
        for stage, lat in latencies.items():
            if lat["n"]:
                print(f"        latency {stage:>5}: avg {lat['avg_ms']} ms, p95 {lat['p95_ms']}"
                      f" ms (n={lat['n']} turns)")


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["text", "audio"], default="text")
    parser.add_argument("--dry-run", action="store_true", help="fake speech client, no network")
    parser.add_argument("--limit", type=int, default=None, help="only the first N calls")
    parser.add_argument("--calls", type=Path, default=CALLS)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--stt-model", default=None, help="override MODEL_STT")
    parser.add_argument("--tts-model", default=None, help="override MODEL_TTS")
    parser.add_argument("--tag", default=None, help="added to the run name, e.g. before-fixes")
    args = parser.parse_args(argv)

    settings = load_settings()
    overrides = {k: v for k, v in (("stt_model", args.stt_model), ("tts_model", args.tts_model))
                 if v}
    settings = dataclasses.replace(settings, **overrides)
    today = date.today().isoformat()

    if args.dry_run:
        out_dir = args.out or EVALS / "dry_run"
        run_id, label = f"{args.mode}_dry_run_{today}", DRY_RUN_LABEL
        audio_client = FakeAudioClient()
    elif args.mode == "text":
        out_dir = args.out or EVALS / "results"
        run_id, label = f"text_only_{today}", TEXT_ONLY_LABEL
        audio_client = None
    else:
        out_dir = args.out or EVALS / "results"
        slug = settings.stt_model.replace("/", "_")
        run_id = f"audio_{slug}_{today}"
        label = f"live audio run: STT {settings.stt_model}, TTS {settings.tts_model}"
        try:
            audio_client = OpenRouterAudioClient(settings, trace_path=out_dir / "traces.jsonl")
        except MissingKeyError as error:
            sys.exit(f"{error} Use --dry-run to test the pipeline without a key.")
        if args.limit:
            run_id += f"_first{args.limit}"  # a smoke run never overwrites the full run
        if args.tag:
            run_id += f"_{args.tag}"

    tools = make_order_tools(settings.order_api_url)
    rows, all_turns, total_cost = [], [], 0.0
    for call in load_calls(args.calls, args.limit):
        row, turns = run_call(call, args.mode, tools, audio_client, args.dry_run)
        rows.append(row)
        all_turns.extend(turns)
        total_cost += row["cost_usd"] or 0.0
        if total_cost > settings.max_cost_per_run_usd:  # budget guard from AGENTS.md
            print(f"Stopping: cost US${total_cost:.2f} passed MAX_COST_PER_RUN_USD.")
            break

    extra = {}
    if args.mode == "audio" and not args.dry_run:
        cost_sources = add_exact_costs(rows, all_turns, audio_client)
        extra = {"tts_voice": settings.tts_voice,
                 "cost_source": "STT: usage.cost in each response; TTS: OpenRouter "
                                "GET /api/v1/generation total_cost per reply, or characters "
                                "x per-character price when that record is missing",
                 "tts_cost_sources": cost_sources,
                 "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 6),
                 "note": "Caller audio is synthetic TTS (clean studio-like speech, plus white "
                         "noise on 2 calls), not real phone audio. Callers are scripted and "
                         "do not adapt when STT mishears."}
    summary = {"label": label, "run_id": run_id, "date": today, "mode": args.mode,
               "stt_model": None if args.mode == "text" else audio_client.stt_model,
               "tts_model": None if args.mode == "text" else audio_client.tts_model,
               **extra, **summarise(rows, all_turns)}
    write_outputs(out_dir, run_id, rows, all_turns, summary)
    print_summary(summary)
    print(f"  wrote {out_dir}/{run_id}_*")
    return summary


if __name__ == "__main__":
    main()
