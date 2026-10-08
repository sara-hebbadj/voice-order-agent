# Version 2 (optional, not built): live real-time voice

Status on 8 October 2026: **design notes only.** Nothing in this repository calls a
real-time API yet, and no version 2 numbers exist.

## Why consider it

Version 1 is turn-based: the caller finishes speaking, then speech-to-text, the agent and
text-to-speech run one after another, so the caller waits for the sum of three stages.
A speech-to-speech real-time model streams audio both ways, detects the end of the
caller's turn itself and can start answering within one connection, so the caller should
hear the first audio sooner. It also handles barge-in (the caller interrupting).

## Model and access

- The build spec names OpenAI's `gpt-realtime-2.1` and `gpt-realtime-2.1-mini`
  (reported as released on 6 July 2026; check OpenAI's model page before building).
  Use the **mini** model for cost.
- It runs on the OpenAI Realtime API, so it needs `OPENAI_API_KEY`. OpenRouter's audio
  endpoints used in version 1 are request/response only.

## Plan

1. **Connection.** Server side: a WebSocket session to the Realtime API. Browser demo:
   WebRTC with a short-lived client token minted by our server (never ship the API key
   to the browser).
2. **Session settings.** Instructions (the same rules as version 1: Arabic or English,
   read order numbers back digit by digit, no details before verification, offer a
   person), a voice, server-side turn detection, and input transcription switched on so
   we still get a transcript for logs and scoring.
3. **Tools.** Expose the same operations as function tools:
   `verify_and_get_order(order_id, phone_last4)`, `get_tracking(tracking_id)` and
   `handover_to_human(reason)`. Important: **verification stays in our code**, inside the
   tool. The tool returns order data only when the phone digits match, so the model can
   never read out an unverified order even if it is talked into trying.
4. **Measure on the same 30 calls.** Stream each synthesised WAV from `evals/audio/` into
   the session at real-time speed and record: time from the end of the caller's audio
   to the first agent audio byte, task success, order-number capture, handover decisions
   and cost per call (Realtime usage reports audio tokens).
5. **Compare with version 1.** For a fair latency comparison, also add "time to first
   audio byte" to version 1 by streaming the TTS response, instead of comparing v2's
   first byte with v1's full turn.

## Trade-offs to discuss

| | Version 1: turn-based (built) | Version 2: real-time (planned) |
|---|---|---|
| Latency | sum of STT + agent + TTS per turn | streaming; expected to be lower (to be measured) |
| Control | every step is visible and testable; agent is deterministic | the model decides what to say; guard rails must live in tools |
| Cost | pay per audio second (STT) + per character (TTS) | pay for audio tokens in and out for the whole session |
| Debugging | transcript, tool calls and timing per stage | one stream; needs event logging |
| Vendor | any OpenRouter STT/TTS model | OpenAI Realtime API only |
