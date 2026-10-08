# LEARN: voice-order-agent

For Sara: a 10-minute walkthrough you can give in an interview, 10 likely questions with
short answers, and 3 small changes to practise live.

## 10-minute walkthrough script

1. **The problem (1 min).** "Order-status calls are the most common call for an online
   shop. I built a voice bot that answers them in Arabic or English, but only after it
   has confirmed the order number and checked the caller, and it passes the caller to a
   person when needed." Show the README's first section.
2. **Demo (2 min).** Run `python app/app.py`. Type (or say) `My order is LS-10154`, answer
   `yes`, then the last 4 digits from the "Demo orders" panel. Point at the transcript,
   the two tool calls and the latency table. Then start a new call and say
   `أبي أكلم موظف` to show the handover.
3. **The pipeline (2 min).** Open `src/voice_agent/pipeline.py`: three stages, each timed:
   speech-to-text, agent, text-to-speech. Open `audio_client.py`: the real OpenRouter client
   and the fake client used by the tests, with the same two methods.
4. **The agent (2 min).** Open `agent.py`: the stages `ask_order → confirm_order →
   ask_phone → done`, plus `handover` and `ended`. Show `_verify`: no status is read until
   the order number and the last 4 phone digits match, and the same message is used for
   "unknown order" and "wrong phone".
5. **The parser (1 min).** Open `parser.py` and `tests/test_parser.py`: Arabic-Indic digits,
   digit words in both languages, "double zero", and why a numeral and a digit word never
   join ("the last four 7669").
6. **Evaluation (2 min).** Open `evals/calls.jsonl` (30 calls, 15 per language) and
   `evals/scoring.py`. Run `python -m evals.run --mode text`. Explain why the text-only
   30/30 is a consistency check, not a speech result, and what the live audio run will
   measure (per-stage latency, capture accuracy, cost per call).

## 10 interview questions with short answers

1. **Where does the latency come from, and how would you cut it?**
   In version 1 the caller waits for speech-to-text, then the agent, then text-to-speech.
   The agent took about 1 ms per turn in the text-only run, so the time is in the two
   network calls. To cut it: stream the TTS audio, cache fixed phrases like the
   greeting, choose faster STT/TTS models, keep the agent rule-based, or move to a
   real-time speech-to-speech model.

2. **How do you confirm an order number by voice reliably?**
   Read it back digit by digit and ask yes/no; accept corrections ("no, it's 10054");
   check the length (five digits); ask for digits one by one when the caller says it as
   a quantity; and never act on it until the phone digits also match.

3. **Why is the agent rule-based instead of an LLM?**
   Verification must be exact and testable, the replies must never include personal
   data, and every LLM call adds latency to each turn. The model calls are where they
   add the most value: speech recognition and speech synthesis.

4. **When should a voice bot hand over to a human?**
   When the caller asks (any wording, any stage, both languages), after two failed tries,
   and for anything outside its job (for example a refund question after a return). It
   passes a ticket (reason, language, order number, verified or not) so the caller does
   not repeat themselves.

5. **How do you stop the bot leaking someone else's order?**
   Details only after the order number and the last 4 phone digits match; the same
   failure message whether the order exists or not; the API returns only the last 4
   digits; replies never include name, address or amount; two failures end in handover.

6. **How do you handle Arabic and English in one system?**
   Language is detected from the script of the transcript (Arabic-Indic digits count as
   Arabic). The first utterance with letters sets it, and a one-word reply like "OK"
   does not switch it. All replies are written in both languages, and Arabic digit words
   are normalised (hamza, taa marbuta) so one keyword list covers spelling variants.

7. **How did you evaluate it without real customers?**
   30 scripted calls (15 Arabic, 15 English) covering happy paths, corrections, wrong
   phone digits, unknown orders, requests for a person, noise and fast speech. The audio
   is generated with TTS in different voices. Metrics: task success, order-number
   capture, correct handovers, status leaks, latency per stage and cost per call.

8. **What does the text-only 30/30 actually prove?**
   That the dialogue logic and scoring are consistent on the scripts. It does not prove
   speech accuracy, because the scripts and the parser were written together. The live
   audio run is the real test, and scripted callers make it a lower bound.

9. **How do you keep tests free of network calls and secrets?**
   Every model call goes through one client with the same interface as a fake client.
   Tests use the fake, or a mock HTTP transport to check the exact request shape. CI runs
   pytest and ruff with no secrets.

10. **Turn-based or real-time voice: which would you ship?**
    Turn-based is cheaper, easier to debug and fully controllable; real-time feels more
    natural and supports interruptions but puts more decisions in the model. I would
    measure both on the same 30 calls, and keep verification inside a tool either way.

## 3 "change it live" exercises

1. **Allow three tries instead of two.** In `agent.py` change `MAX_FAILURES = 2` to `3`.
   Run `pytest -q`: `test_two_failed_verifications_hand_over` should fail. Update the test
   and rebuild the expectations (`evals/build_calls.py` cases 8 and 10 expect a handover
   after two failures), then run `python -m evals.run --mode text` and explain the change.
2. **Add a new way to ask for a person.** Make the Gulf phrase "أبي أحد يساعدني" hand over.
   First try the obvious fix: add `"احد"` to the `handover` words in `parser.py` and run
   `python -m evals.run --mode text`. Calls that say the digit "واحد" (one) now hand over.
   Why? (`_arabic_variants` strips the leading "و" from "واحد", leaving "احد".) Then do it
   properly: add the phrase `"احد يساعدني"` to `INTENT_PHRASES["handover"]` instead, add a
   test in `tests/test_parser.py`, and rerun the evaluation.
3. **Mention a failed delivery attempt.** In the data, 19 delivered orders had a failed
   delivery attempt before delivery. In `replies.py`, when a delivered order's tracking
   events include `failed_attempt`, add "The courier needed a second attempt." in English
   and Arabic. Find such an order with
   `grep failed_attempt data/tracking_events.csv | head -1`, look up its order in
   `data/orders.csv`, and add a test in `tests/test_agent.py`.
