"""Run the 30 scripted test calls through the agent and score them.

  python -m evals.run --mode text              # now: scripts as text, no speech (real numbers)
  python -m evals.run --mode audio --dry-run   # fake speech client, proves the pipeline
  python -m evals.synthesize_audio             # once a key exists: make the test-call audio
  python -m evals.run --mode audio             # live: STT -> agent -> TTS per turn

Outputs go to evals/results/ (real runs) or evals/dry_run/ (never real results):
  <run>_calls.csv, <run>_turns.jsonl, <run>_summary.json, and traces.jsonl for audio runs.
"""

import argparse
import csv
import dataclasses
import json
import sys
from datetime import date
from pathlib import Path

from evals.scoring import score_call, summarise
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
        if mode == "text":
            agent_turn, timings = run_text_turn(agent, script)
            heard = script
        else:
            audio, filename = audio_for_turn(call, index, dry_run)
            voice_turn = run_voice_turn(agent, audio_client, audio, filename=filename)
            agent_turn, timings = voice_turn.agent_turn, voice_turn.timings_ms
            heard = voice_turn.transcript.text
            cost += voice_turn.transcript.usage.get("cost") or 0.0
            cost += _tts_cost(audio_client, agent_turn.reply)
        turns.append({
            "call_id": call["call_id"], "turn": index + 1, "script": script, "heard": heard,
            "reply": agent_turn.reply, "stage": agent_turn.stage,
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


def _tts_cost(audio_client, text: str) -> float:
    price = getattr(getattr(audio_client, "settings", None), "tts_price_per_1m_chars", None)
    return len(text) * price / 1e6 if price else 0.0


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
    for stage, lat in summary["latency_per_turn"].items():
        if lat["n"]:
            print(f"  latency {stage:>5}: avg {lat['avg_ms']} ms, p95 {lat['p95_ms']} ms "
                  f"(n={lat['n']} turns)")


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["text", "audio"], default="text")
    parser.add_argument("--dry-run", action="store_true", help="fake speech client, no network")
    parser.add_argument("--limit", type=int, default=None, help="only the first N calls")
    parser.add_argument("--calls", type=Path, default=CALLS)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--stt-model", default=None, help="override MODEL_STT")
    parser.add_argument("--tts-model", default=None, help="override MODEL_TTS")
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

    summary = {"label": label, "run_id": run_id, "date": today, "mode": args.mode,
               "stt_model": None if args.mode == "text" else audio_client.stt_model,
               "tts_model": None if args.mode == "text" else audio_client.tts_model,
               **summarise(rows, all_turns)}
    write_outputs(out_dir, run_id, rows, all_turns, summary)
    print_summary(summary)
    print(f"  wrote {out_dir}/{run_id}_*")
    return summary


if __name__ == "__main__":
    main()
