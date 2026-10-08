"""Gradio demo: speak (or type) to the Lumi Skin order-status agent.

  python app/app.py        then open http://127.0.0.1:7860

Voice needs OPENROUTER_API_KEY (speech-to-text + text-to-speech). Typing works without
a key, so the agent logic can be tried offline. All data is synthetic.
"""

import csv
import tempfile
from pathlib import Path

import gradio as gr

from voice_agent.agent import OrderStatusAgent
from voice_agent.audio_client import MissingKeyError, OpenRouterAudioClient
from voice_agent.config import DATA_DIR, load_settings
from voice_agent.pipeline import run_text_turn, run_voice_turn
from voice_agent.tools import make_order_tools

MAX_TURNS_PER_CALL = 20  # demo limit, because every spoken turn costs money
LATENCY_HEADERS = ["turn", "input", "stt_ms", "agent_ms", "tts_ms", "total_ms"]

settings = load_settings()
tools = make_order_tools(settings.order_api_url)
try:
    audio_client = OpenRouterAudioClient(settings)
except MissingKeyError:
    audio_client = None


def demo_orders(n_per_status: int = 1) -> list[list[str]]:
    """A few synthetic orders and their phone digits, so visitors can try the demo."""
    with (DATA_DIR / "customers.csv").open(encoding="utf-8") as f:
        last4 = {c["id"]: c["phone"][-4:] for c in csv.DictReader(f)}
    rows, seen = [], {}
    with (DATA_DIR / "orders.csv").open(encoding="utf-8") as f:
        for order in csv.DictReader(f):
            if seen.get(order["status"], 0) < n_per_status:
                seen[order["status"]] = seen.get(order["status"], 0) + 1
                rows.append([order["order_id"], order["status"], last4[order["customer_id"]]])
    return rows


def new_call() -> tuple:
    session = {"agent": OrderStatusAgent(tools), "latency": [], "turns": 0}
    chat = [{"role": "assistant", "content": OrderStatusAgent.greeting()}]
    return session, chat, None, [], []


def _record(session: dict, chat: list, heard: str, agent_turn, timings: dict, kind: str):
    session["turns"] += 1
    chat = chat + [{"role": "user", "content": heard or "(nothing heard)"},
                   {"role": "assistant", "content": agent_turn.reply}]
    session["latency"].append([session["turns"], kind, timings["stt"], timings["agent"],
                               timings["tts"], timings["total"]])
    calls = [{"tool": c.name, "args": c.args, "ok": c.ok, "latency_ms": round(c.latency_ms, 1)}
             for c in agent_turn.tool_calls]
    return chat, calls


def on_voice(audio_path: str | None, session: dict, chat: list):
    if session is None:
        session, chat, *_ = new_call()
    if not audio_path:
        return session, chat, None, gr.skip(), session["latency"]
    if audio_client is None:
        gr.Warning("Voice needs OPENROUTER_API_KEY in .env. You can type instead.")
        return session, chat, None, gr.skip(), session["latency"]
    if session["turns"] >= MAX_TURNS_PER_CALL:
        gr.Warning("Demo limit reached. Press 'New call'.")
        return session, chat, None, gr.skip(), session["latency"]

    audio = Path(audio_path).read_bytes()
    turn = run_voice_turn(session["agent"], audio_client, audio, filename=Path(audio_path).name)
    chat, calls = _record(session, chat, turn.transcript.text, turn.agent_turn,
                          turn.timings_ms, "voice")
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(turn.reply_audio)
    return session, chat, f.name, calls, session["latency"]


def on_text(text: str, session: dict, chat: list):
    if session is None:
        session, chat, *_ = new_call()
    if not text.strip() or session["turns"] >= MAX_TURNS_PER_CALL:
        return session, chat, "", gr.skip(), session["latency"]
    agent_turn, timings = run_text_turn(session["agent"], text)
    chat, calls = _record(session, chat, text, agent_turn, timings, "text")
    return session, chat, "", calls, session["latency"]


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Lumi Skin voice order agent") as demo:
        gr.Markdown(
            "# Lumi Skin order-status voice agent (Arabic / English)\n"
            "An automated assistant for a **fictional** shop with **synthetic** data. "
            "Say or type an order number, confirm it, then give the last 4 phone digits. "
            "Ask for a person at any time.\n\n"
            + ("Voice is on." if audio_client else
               "**Voice is off** (no OPENROUTER_API_KEY). Typing still works.")
        )
        session = gr.State(None)
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(label="Transcript", height=420)
                mic = gr.Audio(sources=["microphone", "upload"], type="filepath",
                               label="Speak (Arabic or English)")
                text = gr.Textbox(label="Or type", placeholder="My order is LS-10154")
                with gr.Row():
                    send_voice = gr.Button("Send voice", variant="primary")
                    send_text = gr.Button("Send text")
                    reset = gr.Button("New call")
            with gr.Column(scale=2):
                reply_audio = gr.Audio(label="Agent reply", autoplay=True, interactive=False)
                tool_calls = gr.JSON(label="Tool calls (last turn)")
                latency = gr.Dataframe(headers=LATENCY_HEADERS, label="Latency per turn (ms)")
                with gr.Accordion("Demo orders (synthetic)", open=False):
                    gr.Dataframe(value=demo_orders(),
                                 headers=["order", "status", "phone last 4"])

        send_voice.click(on_voice, [mic, session, chat],
                         [session, chat, reply_audio, tool_calls, latency])
        send_text.click(on_text, [text, session, chat],
                        [session, chat, text, tool_calls, latency])
        text.submit(on_text, [text, session, chat], [session, chat, text, tool_calls, latency])
        reset.click(new_call, None, [session, chat, reply_audio, tool_calls, latency])
        demo.load(new_call, None, [session, chat, reply_audio, tool_calls, latency])
    return demo


if __name__ == "__main__":
    build_ui().launch()
