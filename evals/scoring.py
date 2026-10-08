"""Score one test call, and summarise many.

Definitions (also in the README):
- task success: a "status" call ends with the right status for the right order and no
  handover; a "handover" call ends in a handover for the expected reason with no status read.
- order-number capture: the order number the caller confirmed equals the expected one
  (only for calls that have an expected order number).
- correct handover decision: handed over if and only if expected, for the expected reason.
- status leak: an order status was read out in a call that should end in a handover.
- language: the agent's last reply is in the caller's language.

Speech-to-text accuracy, per caller turn (audio runs only), comparing what STT heard
with the script the caller audio was made from:
- word error rate (WER): word edits / words in the script, after normalising both sides
  (case, punctuation, Arabic letter variants) and splitting every number into single
  digits, so "10154" and "one zero one five four" count as the same five words;
- number capture: the digit groups the parser finds in the transcript equal the ones it
  finds in the script (turns whose script contains digits);
- order-ID capture: the 5-digit order number parsed from the transcript equals the one in
  the script (turns whose script contains an order number).
"""

import numpy as np

from voice_agent.parser import comparable_words, extract_digit_groups, find_order_number


def word_edits(reference: list[str], hypothesis: list[str]) -> int:
    """Levenshtein distance between two word lists (substitutions, insertions, deletions)."""
    previous = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, start=1):
        current = [i]
        for j, hyp_word in enumerate(hypothesis, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ref_word != hyp_word)))
        previous = current
    return previous[-1]


def score_turn(script: str, heard: str) -> dict:
    """How well speech-to-text preserved one caller turn (see the module docstring)."""
    reference, hypothesis = comparable_words(script), comparable_words(heard)
    script_groups, heard_groups = extract_digit_groups(script), extract_digit_groups(heard)
    script_order = find_order_number(script_groups)
    return {
        "ref_words": len(reference),
        "word_edits": word_edits(reference, hypothesis),
        "numbers_ok": (heard_groups == script_groups) if script_groups else None,
        "order_id_ok": (find_order_number(heard_groups) == script_order)
        if script_order else None,
    }


def _digits(order_id: str | None) -> str | None:
    return "".join(ch for ch in order_id if ch.isdigit()) if order_id else None


def score_call(call: dict, outcome: dict) -> dict:
    """`outcome` holds the agent's final state: reported, handover_reason,
    confirmed_order (digits) and language."""
    expected_handover = call["expected_outcome"] == "handover"
    reported = outcome["reported"]  # list of (order_id, status)
    handed_over = outcome["handover_reason"] is not None

    if expected_handover:
        task_success = (outcome["handover_reason"] == call["expected_handover_reason"]
                        and not reported)
    else:
        expected = (call["expected_order_id"], call["expected_status"])
        task_success = not handed_over and bool(reported) and tuple(reported[-1]) == expected

    expected_digits = _digits(call["expected_order_id"])
    capture_ok = None if expected_digits is None else outcome["confirmed_order"] == expected_digits
    handover_ok = handed_over == expected_handover and (
        not expected_handover or outcome["handover_reason"] == call["expected_handover_reason"]
    )
    return {
        "task_success": task_success,
        "capture_ok": capture_ok,
        "handover_ok": handover_ok,
        "status_leak": expected_handover and bool(reported),
        "language_ok": outcome["language"] == call["language"],
    }


def _rate(values: list) -> dict:
    values = [v for v in values if v is not None]
    k = sum(bool(v) for v in values)
    n = len(values)
    return {"k": k, "n": n, "pct": round(100 * k / n, 1) if n else None}


def _latency(values: list) -> dict:
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0, "avg_ms": None, "p95_ms": None}
    return {"n": len(values), "avg_ms": round(float(np.mean(values)), 2),
            "p95_ms": round(float(np.percentile(values, 95)), 2)}


def summarise(rows: list[dict], turns: list[dict]) -> dict:
    """rows: one scored dict per call; turns: one dict per caller turn with timings_ms."""
    summary = {"calls": len(rows)}
    for lang in ("all", "en", "ar"):
        sub = [r for r in rows if lang == "all" or r["language"] == lang]
        expected_handovers = [r for r in sub if r["expected_outcome"] == "handover"]
        summary[lang] = {
            "task_success": _rate([r["task_success"] for r in sub]),
            "order_capture": _rate([r["capture_ok"] for r in sub]),
            "handover_decision": _rate([r["handover_ok"] for r in sub]),
            "handover_recall": _rate([r["handover_ok"] for r in expected_handovers]),
            "false_handovers": sum(1 for r in sub if r["expected_outcome"] == "status"
                                   and r["handover_reason"] is not None),
            "status_leaks": sum(1 for r in sub if r["status_leak"]),
            "language_ok": _rate([r["language_ok"] for r in sub]),
        }
    summary["latency_per_turn"] = _latencies(turns)
    costs = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
    summary["cost_per_call_usd"] = round(float(np.mean(costs)), 5) if costs else None
    if any("stt_scores" in t for t in turns):  # audio runs only
        for lang in ("en", "ar"):
            lang_turns = [t for t in turns if t.get("language") == lang]
            lang_costs = [r["cost_usd"] for r in rows
                          if r["language"] == lang and r.get("cost_usd") is not None]
            summary[lang]["latency_per_turn"] = _latencies(lang_turns)
            summary[lang]["cost_per_call_usd"] = (round(float(np.mean(lang_costs)), 5)
                                                  if lang_costs else None)
        for lang in ("all", "en", "ar"):
            lang_turns = [t for t in turns if lang == "all" or t.get("language") == lang]
            summary[lang]["stt"] = _stt_accuracy([t["stt_scores"] for t in lang_turns])
    return summary


def _latencies(turns: list[dict]) -> dict:
    return {stage: _latency([t["timings_ms"][stage] for t in turns])
            for stage in ("stt", "agent", "tts", "total")}


def _stt_accuracy(scores: list[dict]) -> dict:
    ref_words = sum(s["ref_words"] for s in scores)
    edits = sum(s["word_edits"] for s in scores)
    return {
        "turns": len(scores),
        "wer": {"edits": edits, "ref_words": ref_words,
                "pct": round(100 * edits / ref_words, 1) if ref_words else None},
        "number_capture": _rate([s["numbers_ok"] for s in scores]),
        "order_id_capture": _rate([s["order_id_ok"] for s in scores]),
    }
