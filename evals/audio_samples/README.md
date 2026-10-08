# Sample clips (about 1.4 MB)

A few clips from the live evaluation on 8 October 2026, so you can hear what was tested
without running anything. The full test audio (116 caller clips, about 15 MB) and every
agent reply are git-ignored; recreate them with `python -m evals.synthesize_audio` and
`python -m evals.run --mode audio`.

All voices are synthetic (text-to-speech through OpenRouter); no real person's voice.

| File | What it is |
|---|---|
| `en01_caller_turn1.wav` | Test caller (google/gemini-3.8-flash-lite-tts): "Hi, my order number is LS-10154." |
| `en01_agent_reply1.mp3` | Agent (elevenlabs/eleven-flash-v2.5, voice "sarah") reads the number back digit by digit |
| `en01_agent_reply3_status.mp3` | Agent gives the order status after the phone check |
| `ar02_caller_turn1.wav` | Arabic caller says the order number as digit words: "واحد صفر صفر واحد وواحد" |
| `ar02_agent_reply1.mp3` | Agent reads it back in Arabic |
| `ar02_caller_turn3.wav` | Arabic caller gives the last 4 phone digits as words |
| `ar02_agent_reply3_status.mp3` | Agent gives the status in Arabic |
| `en12_caller_turn1_fast_noisy.wav` | Fast speech with white noise at 10 dB SNR. openai/gpt-4o-transcribe turned it into Polish or Swedish text; elevenlabs/scribe-v2 got it right |
| `en07_caller_turn1_REJECTED_tts_said_1004.wav` | A **rejected** test clip: the script says "10004" but the TTS voice said "one zero zero four". All four STT models heard 1004, so the clip was re-made (see `../results/stt_crosscheck_2026-10-08.jsonl`) |

The Arabic agent voice has not been reviewed by a native speaker yet.
