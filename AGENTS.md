# Notes for coding agents (voice-order-agent)

Read the README first. This is a portfolio project: Sara must be able to explain every
line, so keep functions short, names plain and comments on non-obvious decisions.

## Layout

- `src/voice_agent/`: `config.py` (settings from `.env`), `audio_client.py` (OpenRouter
  STT/TTS + `FakeAudioClient`), `parser.py` (rules: digits, intents, language),
  `agent.py` (state machine), `replies.py` (all spoken text, EN + AR), `tools.py`
  (HTTP tools), `mock_api.py` (FastAPI mock of the companion project's API),
  `pipeline.py` (one timed turn).
- `app/app.py`: Gradio demo. `evals/`: test calls, runner, scoring, audio synthesis.

## Rules

- **No network in tests.** Use `FakeAudioClient` or a mock HTTP transport. CI installs
  `.[dev]` only (no gradio), so `tests/test_app.py` is skipped there.
- **Secrets:** read only through `config.load_settings()`. Never print, log or trace a key.
- **The agent stays deterministic.** Verification (order number + last 4 phone digits)
  and the "no details before verification" rule live in code, not in a prompt. If you add
  an LLM, use it only as a fallback for turns the parser cannot understand, keep it behind
  the same client pattern with a fake, and measure it.
- **Same message** for "order not found" and "phone does not match" (no enumeration).
- **Replies** go in `replies.py` in both languages. Read digits one by one via
  `spell_digits`. Never speak names, addresses, amounts or full phone numbers.
- **Evaluation honesty:** real runs write to `evals/results/`; `--dry-run` writes to
  `evals/dry_run/` and is labelled NOT real. Never copy a number into the README or
  RESULTS.md unless a saved run produced it (state the denominator, model IDs and date).
- If `data/*.csv` change, run `python -m evals.build_calls` (a test checks that
  `evals/calls.jsonl` matches the data).

## Compatibility with the companion project (P1)

The tools call `GET /orders/{id}` and `GET /tracking/{tracking_id}`, the same paths as
P1's mock API. This repo's `get_order` response includes `customer_phone_last4`; if P1's
response differs, adapt `OrderStatusAgent._verify` and `replies.status_reply`, or point
`ORDER_API_URL` only at a server that returns that field.

## Commands

```bash
pytest -q && ruff check .
python -m evals.run --mode text
python -m evals.run --mode audio --dry-run
python app/app.py
```

Live audio (costs money; read the README "How to run" first): `evals.synthesize_audio`,
then `evals.crosscheck_stt --all-with-digits` to check the test audio, then
`evals.run --mode audio --tag <name>` (each live run is about US$0.19; give each run its
own tag so files are not overwritten), and `evals.recost` for late TTS costs.
