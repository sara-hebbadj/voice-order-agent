# Architecture

## Version 1: turn-based voice (built)

```mermaid
flowchart LR
  M[Mic in browser<br>Gradio] -->|WAV| STT[Speech-to-text<br>OpenRouter /audio/transcriptions]
  STT -->|transcript| P[Parser<br>digits, intents, language]
  P --> A[Agent state machine<br>confirm, verify, reply]
  A -->|GET /orders/id<br>GET /tracking/id| API[Orders + courier mock API<br>FastAPI, synthetic data]
  A -->|reply text AR/EN| TTS[Text-to-speech<br>OpenRouter /audio/speech]
  TTS -->|MP3| S[Speaker]
  A -->|handover ticket| H[Human colleague]
  STT -.-> T[(traces.jsonl<br>model, latency, cost)]
  TTS -.-> T
```

Each spoken turn runs three stages in order, and each is timed separately
(`src/voice_agent/pipeline.py`):

1. **Speech-to-text** (`audio_client.py`): the caller's audio goes to OpenRouter's
   OpenAI-compatible transcription endpoint (multipart upload). Default model:
   `elevenlabs/scribe-v2` (set `MODEL_STT`), chosen by comparing 4 models on the 30 test
   calls (README, Results). From the second turn on, the call's language is sent as a
   hint (`pipeline.stt_language_hint`): without it, one-word answers such as "نعم" came
   back in the wrong language. `openai/gpt-4o-transcribe` ignored the hint, the other
   three models we tried followed it.
2. **Agent** (`parser.py`, `agent.py`, `tools.py`, `replies.py`): deterministic rules, no
   LLM. The parser turns the transcript into digits, intents and a language guess. The
   agent is a state machine:

```mermaid
stateDiagram-v2
  [*] --> ask_order: bilingual greeting + AI disclosure
  ask_order --> confirm_order: 5-digit number heard
  confirm_order --> confirm_order: correction ("no, it's 10054")
  confirm_order --> ask_order: "no"
  confirm_order --> ask_phone: "yes"
  ask_phone --> done: order + last 4 phone digits match (status read out)
  ask_phone --> ask_order: no match (same message for unknown order / wrong phone)
  done --> confirm_order: another order (verify again)
  done --> ended: "no, thanks" / goodbye
  ask_order --> handover: 2 failed tries
  ask_phone --> handover: 2 failed tries
  ask_order --> handover: asks for a person (any state)
```

3. **Text-to-speech**: the reply goes to OpenRouter's speech endpoint. Default model:
   `elevenlabs/eleven-flash-v2.5`, voice `sarah` (set `MODEL_TTS`, `TTS_VOICE`): the
   fastest of three models tried (avg 426 ms per reply in the final live run), priced per
   character. Delivery instructions (calm, read digits slowly) are sent too; OpenRouter
   forwards them only to OpenAI and Gemini voices, so this voice ignores them. The
   speech endpoint returns only audio, so the evaluation asks OpenRouter for each
   reply's cost by its generation ID (`audio_client.generation_costs`).

The test callers use a different voice model, `google/gemini-3.8-flash-lite-tts`
(`MODEL_TTS_CALLER`, only in `evals/synthesize_audio.py`), which follows instructions such
as "speak quickly".

### Why the agent has no LLM

- Order numbers and phone digits must be exact; rules are testable and auditable.
- An LLM call per turn would add a fourth network round trip to every spoken turn.
- The spoken replies are fixed templates, so nothing personal (name, address, amount)
  can be said by accident, and the Arabic wording was written, not generated.

The trade-off: the parser only understands the phrasings it was written for. A caller who
says "ten thousand and forty-five" is asked to say the digits one by one. The live runs
showed what real transcripts look like, and the parser now also handles "LS10154" (no
hyphen), commas between digit words ("ثمانية، ثلاثة"), a phone ending written as a time
("12:41"), and digits in English words inside an Arabic call. Numbers said in pairs or as
quantities ("سبعة وعشرين تسعين") are still not understood (`xfail` tests in
`tests/test_parser.py`). An LLM-based fallback for unclear turns is listed under next
steps in the README.

### Safety and privacy decisions

- Details only after the order number **and** the last 4 phone digits match.
- The same "couldn't match" message for an unknown order and a wrong phone, so the bot
  cannot be used to discover which order numbers exist.
- The mock API exposes only the last 4 phone digits (data minimisation).
- `hmac.compare_digest` for the digit check (constant time).
- Two failed tries → handover; a request for a person → handover at any point, with a
  ticket (reason, language, order number, verified yes/no) so the caller need not repeat.
- The greeting says the caller is talking to an automated assistant.

## Version 2: live real-time voice (not built)

See [realtime_v2.md](realtime_v2.md).
