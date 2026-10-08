import json
import wave
from collections import Counter
from io import BytesIO

import numpy as np

from evals import run, synthesize_audio
from evals.build_calls import build
from evals.scoring import score_call


def load_committed_calls():
    return run.load_calls(run.CALLS, None)


def test_thirty_calls_fifteen_per_language():
    calls = load_committed_calls()
    assert len(calls) == 30
    assert Counter(c["language"] for c in calls) == {"en": 15, "ar": 15}
    assert len({c["call_id"] for c in calls}) == 30


def test_committed_calls_match_the_builder_and_the_data():
    # If data/ changes, rebuild with `python -m evals.build_calls`.
    assert load_committed_calls() == build()


def test_text_run_writes_results(tmp_path):
    summary = run.main(["--mode", "text", "--limit", "4", "--out", str(tmp_path)])
    assert summary["calls"] == 4 and summary["label"] == run.TEXT_ONLY_LABEL
    assert list(tmp_path.glob("text_only_*_calls.csv"))


def test_audio_dry_run_is_labelled_and_separate(tmp_path):
    summary = run.main(["--mode", "audio", "--dry-run", "--limit", "3", "--out", str(tmp_path)])
    assert "NOT real results" in summary["label"]
    saved = json.loads(next(tmp_path.glob("*_summary.json")).read_text())
    assert saved["stt_model"] == "fake-stt"
    assert saved["latency_per_turn"]["stt"]["n"] > 0


def test_score_status_call_needs_right_order_and_status():
    call = {"expected_outcome": "status", "expected_status": "shipped",
            "expected_order_id": "LS-10154", "expected_handover_reason": None, "language": "en"}
    good = {"reported": [("LS-10154", "shipped")], "handover_reason": None,
            "confirmed_order": "10154", "language": "en"}
    assert score_call(call, good)["task_success"]
    wrong = good | {"reported": [("LS-10154", "delivered")]}
    assert not score_call(call, wrong)["task_success"]


def test_score_handover_call_flags_a_status_leak():
    call = {"expected_outcome": "handover", "expected_status": None,
            "expected_order_id": "LS-10154", "expected_handover_reason": "repeated_failure",
            "language": "ar"}
    leaked = {"reported": [("LS-10154", "shipped")], "handover_reason": "repeated_failure",
              "confirmed_order": "10154", "language": "ar"}
    scores = score_call(call, leaked)
    assert scores["status_leak"] and not scores["task_success"]


def test_pcm_to_wav_and_noise():
    tone = (8000 * np.sin(np.linspace(0, 200 * np.pi, 24_000))).astype(np.int16).tobytes()
    noisy = np.frombuffer(synthesize_audio.add_noise(tone, snr_db=10), dtype=np.int16)
    clean = np.frombuffer(tone, dtype=np.int16).astype(np.float64)
    noise = noisy.astype(np.float64) - clean
    snr = 10 * np.log10(np.mean(clean**2) / np.mean(noise**2))
    assert 9 < snr < 11
    with wave.open(BytesIO(synthesize_audio.pcm_to_wav(tone))) as wav:
        assert wav.getframerate() == 24_000 and wav.getnframes() == 24_000
