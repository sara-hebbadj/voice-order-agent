"""The order-status agent: a small, rule-based dialogue state machine.

Why rules and not an LLM here? Verification and order lookup must be exact and
auditable, and each LLM call would add latency to every spoken turn. Speech-to-text
and text-to-speech are the model calls in this project (see pipeline.py).

Call flow:
  ask_order -> confirm_order (read digits back) -> ask_phone (last 4 digits)
  -> verify with the API -> status reply -> done (anything else?) -> ended
  A request for a person, at any point, -> handover.
  Two failed tries (no usable number, or details that do not match) -> handover.
"""

import hmac
from dataclasses import dataclass, field

from voice_agent.parser import Parsed, parse
from voice_agent.replies import GREETING, say, spell_digits, status_reply
from voice_agent.tools import OrderTools, ToolCall

MAX_FAILURES = 2

ASK_ORDER = "ask_order"
CONFIRM_ORDER = "confirm_order"
ASK_PHONE = "ask_phone"
DONE = "done"  # status given, waiting for "anything else?"
HANDOVER = "handover"
ENDED = "ended"
FINAL_STAGES = {HANDOVER, ENDED}


@dataclass
class CallState:
    stage: str = ASK_ORDER
    language: str = "en"
    language_known: bool = False
    candidate_order: str | None = None  # heard, not yet confirmed by the caller
    confirmed_order: str | None = None  # read back and confirmed by the caller
    verified_order: str | None = None  # order ID + phone digits matched
    failures: int = 0
    last_reply: str = GREETING
    reported: list[tuple[str, str]] = field(default_factory=list)  # (order_id, status)
    handover_reason: str | None = None


@dataclass
class AgentTurn:
    reply: str
    language: str
    stage: str
    tool_calls: list[ToolCall]
    parsed: Parsed

    @property
    def ended(self) -> bool:
        return self.stage in FINAL_STAGES


class OrderStatusAgent:
    def __init__(self, tools: OrderTools):
        self.tools = tools
        self.state = CallState()
        self._calls: list[ToolCall] = []  # tool calls made during the current turn

    @staticmethod
    def greeting() -> str:
        return GREETING

    def handle(self, text: str) -> AgentTurn:
        """Process one caller utterance and return the agent's reply."""
        parsed = parse(text)
        self._calls = []
        self._update_language(parsed)
        state = self.state

        if state.stage in FINAL_STAGES:
            reply = state.last_reply  # the call is over; repeat the final message
        elif "handover" in parsed.intents:
            reply = self._handover("requested")
        elif "repeat" in parsed.intents and not parsed.digit_groups:
            reply = state.last_reply
        else:
            handler = {
                ASK_ORDER: self._on_ask_order,
                CONFIRM_ORDER: self._on_confirm_order,
                ASK_PHONE: self._on_ask_phone,
                DONE: self._on_done,
            }[state.stage]
            reply = handler(parsed)

        state.last_reply = reply
        return AgentTurn(reply, state.language, state.stage, self._calls, parsed)

    # ----- one handler per stage -----

    def _on_ask_order(self, p: Parsed) -> str:
        if p.order_number:
            return self._confirm(p.order_number)
        if p.digit_groups:  # digits, but not five of them
            return self._fail("reprompt_order_length")
        if "goodbye" in p.intents and "order_status" not in p.intents:
            return self._end()
        if "order_status" in p.intents:  # "where is my parcel?": no number yet, not a failure
            return say("ask_order", self.state.language)
        return self._fail("reprompt_order")

    def _on_confirm_order(self, p: Parsed) -> str:
        state = self.state
        if p.order_number and p.order_number != state.candidate_order:
            return self._confirm(p.order_number)  # a correction: "no, it's 10054"
        if "no" in p.intents:
            state.candidate_order = None
            state.stage = ASK_ORDER
            return say("ask_order_again", state.language)
        if "yes" in p.intents or p.order_number == state.candidate_order:
            state.confirmed_order = state.candidate_order
            state.stage = ASK_PHONE
            if p.phone_last4:  # "yes, and my phone ends in 7669"
                return self._verify(p.phone_last4)
            return say("ask_phone", state.language)
        digits = spell_digits(state.candidate_order, state.language)
        return self._fail("reprompt_confirm", digits=digits)

    def _on_ask_phone(self, p: Parsed) -> str:
        if p.phone_last4:
            return self._verify(p.phone_last4)
        if p.order_number and p.order_number != self.state.confirmed_order:
            return self._confirm(p.order_number)  # caller switched to another order
        return self._fail("reprompt_phone")

    def _on_done(self, p: Parsed) -> str:
        if p.order_number:  # another order: it must be verified again
            return self._confirm(p.order_number)
        if p.intents & {"goodbye", "no"}:
            return self._end()
        if "yes" in p.intents:
            self.state.stage = ASK_ORDER
            return say("other_order", self.state.language)
        return self._fail("only_order_status")

    # ----- shared steps -----

    def _update_language(self, p: Parsed) -> None:
        """Take the language from the first utterance that has letters, then only switch
        on a clear sentence (2+ words), so a lone 'OK' does not flip an Arabic call."""
        if p.language and (not self.state.language_known or p.word_count >= 2):
            self.state.language = p.language
            self.state.language_known = True

    def _confirm(self, order_number: str) -> str:
        self.state.candidate_order = order_number
        self.state.stage = CONFIRM_ORDER
        digits = spell_digits(order_number, self.state.language)
        return say("confirm_order", self.state.language, digits=digits)

    def _verify(self, phone_last4: str) -> str:
        state = self.state
        order_id = state.confirmed_order
        call = self.tools.get_order(order_id)
        self._calls.append(call)
        order = call.result if call.ok else None
        expected = (order or {}).get("customer_phone_last4") or ""
        # compare_digest takes the same time whether or not the digits match.
        if not expected or not hmac.compare_digest(expected, phone_last4):
            state.candidate_order = None
            state.stage = ASK_ORDER
            return self._fail("verify_failed")

        tracking = None
        if order.get("tracking_id"):
            tracking_call = self.tools.get_tracking(order["tracking_id"])
            self._calls.append(tracking_call)
            tracking = tracking_call.result if tracking_call.ok else None

        text = status_reply(order, tracking, state.language)
        if text is None:  # a status we cannot explain: let a person handle it
            return self._handover("unknown_status")
        state.verified_order = order["order_id"]  # the API's form, e.g. "LS-10045"
        state.reported.append((order["order_id"], order["status"]))
        state.failures = 0  # a fresh start for any follow-up question
        state.stage = DONE
        return f"{text} {say('anything_else', state.language)}"

    def _fail(self, reprompt_key: str, **values) -> str:
        self.state.failures += 1
        if self.state.failures >= MAX_FAILURES:
            return self._handover("repeated_failure")
        return say(reprompt_key, self.state.language, **values)

    def _handover(self, reason: str) -> str:
        """Hand the call to a person, passing on what we know (a mock ticket here)."""
        state = self.state
        state.stage = HANDOVER
        state.handover_reason = reason
        ticket = {
            "reason": reason,
            "language": state.language,
            "order_id": state.confirmed_order,
            "verified": state.verified_order is not None,
        }
        self._calls.append(ToolCall("handover_to_human", ticket, ok=True, latency_ms=0.0))
        key = "handover_requested" if reason == "requested" else "handover_failure"
        return say(key, state.language)

    def _end(self) -> str:
        self.state.stage = ENDED
        return say("goodbye", self.state.language)
