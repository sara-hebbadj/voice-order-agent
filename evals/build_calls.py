"""Build evals/calls.jsonl: 30 scripted test calls (15 English, 15 Arabic).

Each call is a list of caller utterances plus the expected outcome. Order numbers and
phone digits are filled in from data/ so the expected answers match the mock API.
The same 15 situations are written once per language (not machine-translated):

  1 happy path            6 asks for a person at once   11 says the number as a quantity
  2 digits as words       7 asks for a person mid-call  12 fast speech + background noise
  3 corrects the number   8 wrong phone digits twice    13 says goodbye after the status
  4 starts without number 9 wrong phone, then right     14 misses a digit
  5 gives the full phone 10 order that does not exist   15 asks the bot to repeat

Run:  python -m evals.build_calls
"""

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "evals" / "calls.jsonl"

EN_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
AR_WORDS = ["صفر", "واحد", "اثنين", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة"]
UNKNOWN_ORDER = "10999"  # five digits, but not in orders.csv


def load_orders() -> tuple[dict, dict]:
    with (REPO / "data" / "orders.csv").open(encoding="utf-8") as f:
        orders = list(csv.DictReader(f))
    with (REPO / "data" / "customers.csv").open(encoding="utf-8") as f:
        last4 = {c["id"]: c["phone"][-4:] for c in csv.DictReader(f)}
    by_status: dict[str, list[dict]] = {}
    for order in sorted(orders, key=lambda o: o["order_id"]):
        order["phone4"] = last4[order["customer_id"]]
        by_status.setdefault(order["status"], []).append(order)
    by_id = {o["order_id"]: o for o in orders}
    return by_status, by_id


def en_words(digits: str) -> str:
    return " ".join(EN_WORDS[int(d)] for d in digits)


def ar_words(digits: str) -> str:
    return " ".join(AR_WORDS[int(d)] for d in digits)


def ar_indic(digits: str) -> str:
    return digits.translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))


def plus(phone4: str, n: int) -> str:
    """A different 4-digit number (wrong phone digits for the failure cases)."""
    return f"{(int(phone4) + n) % 10000:04d}"


def misheard(digits: str) -> str:
    """Swap the last two digits, a typical mishearing that the caller corrects."""
    swapped = digits[:-2] + digits[-1] + digits[-2]
    return swapped if swapped != digits else digits[:-1] + str((int(digits[-1]) + 1) % 10)


def status_call(call_id, lang, category, order, turns, tags=()):
    return {
        "call_id": call_id, "language": lang, "category": category, "tags": list(tags),
        "turns": turns, "expected_outcome": "status", "expected_status": order["status"],
        "expected_order_id": order["order_id"], "expected_handover_reason": None,
    }


def handover_call(call_id, lang, category, order_id, reason, turns, tags=()):
    return {
        "call_id": call_id, "language": lang, "category": category, "tags": list(tags),
        "turns": turns, "expected_outcome": "handover", "expected_status": None,
        "expected_order_id": order_id, "expected_handover_reason": reason,
    }


def english_calls(s: dict, by_id: dict) -> list[dict]:
    def o(status, i):
        return s[status][i]

    def d(order):
        return order["order_id"][3:]

    a, b, c, e = o("shipped", 0), o("delivered", 0), o("processing", 0), o("shipped", 1)
    f, g, h, i = o("cancelled", 0), o("delivered", 1), o("returned", 0), o("returned", 1)
    k, m, n, p, q = by_id["LS-10062"], o("shipped", 2), o("delivered", 2), o("processing", 1), \
        o("cancelled", 1)
    return [
        status_call("en01", "en", "happy_path", a,
                    [f"Hi, my order number is {a['order_id']}.", "Yes.", a["phone4"]]),
        status_call("en02", "en", "digits_as_words", b,  # LS-100xx, so "double zero"
                    [f"one double zero {en_words(d(b)[3:])}", "Yes, that's right.",
                     en_words(b["phone4"])]),
        status_call("en03", "en", "correction", c,
                    [f"My order is {misheard(d(c))}.", f"No, the order is {d(c)}.", "Yes.",
                     c["phone4"]]),
        status_call("en04", "en", "opener_without_number", e,
                    ["Hello, I'm calling to check where my parcel is.", f"It's {d(e)}.",
                     "Yeah.", f"The last four are {e['phone4']}."]),
        status_call("en05", "en", "full_phone_number", f,
                    [f"Order {d(f)}, please.", "Correct.",
                     f"My number is +971 50 000 {f['phone4']}."]),
        handover_call("en06", "en", "handover_at_start", None, "requested",
                      ["I want to speak to a human, please."]),
        handover_call("en07", "en", "handover_mid_call", g["order_id"], "requested",
                      [d(g), "Yes.", "Actually, can I talk to a real person?"]),
        handover_call("en08", "en", "wrong_phone_twice", h["order_id"], "repeated_failure",
                      [d(h), "Yes.", plus(h["phone4"], 1111), f"It's {d(h)}.", "Yes.",
                       plus(h["phone4"], 2222)]),
        status_call("en09", "en", "wrong_phone_then_right", i,
                    [d(i), "Yes.", plus(i["phone4"], 1111), f"Sorry. {d(i)}.", "Yes.",
                     i["phone4"]]),
        handover_call("en10", "en", "unknown_order", "LS-" + UNKNOWN_ORDER, "repeated_failure",
                      [f"LS-{UNKNOWN_ORDER}", "Yes.", "4821", f"I said {UNKNOWN_ORDER}.",
                       "Yes.", "4821"]),
        status_call("en11", "en", "number_as_quantity", k,
                    ["My order is ten thousand and sixty two.", en_words(d(k)), "Yes.",
                     k["phone4"]]),
        status_call("en12", "en", "fast_and_noisy", m,
                    [f"Where's my order LS {d(m)}?", "yes yes", m["phone4"]],
                    tags=("fast", "noisy")),
        status_call("en13", "en", "goodbye_after_status", n,
                    [f"Order number {d(n)}.", "That's right.", n["phone4"],
                     "No, that's all. Thank you!"]),
        status_call("en14", "en", "missing_digit", p,
                    [f"My order is {d(p)[:4]}.", f"Sorry, {en_words(d(p))}.", "Yes.",
                     p["phone4"]]),
        status_call("en15", "en", "asks_to_repeat", q,
                    [d(q), "Sorry, can you repeat that?", "Yes.", q["phone4"]]),
    ]


def arabic_calls(s: dict, by_id: dict) -> list[dict]:
    def o(status, i):
        return s[status][i]

    def d(order):
        return order["order_id"][3:]

    a, b, c, e = o("shipped", 3), o("delivered", 3), o("processing", 2), o("shipped", 4)
    f, g, h, i = o("cancelled", 2), o("delivered", 4), o("returned", 2), o("returned", 3)
    k, m, n, p, q = by_id["LS-10045"], o("shipped", 5), o("delivered", 5), o("processing", 3), \
        o("cancelled", 3)
    wb = ar_words(d(b)).rsplit(" ", 1)  # say "and" before the last digit, as people do
    return [
        status_call("ar01", "ar", "happy_path", a,
                    [f"مرحبا، رقم طلبي {ar_indic(d(a))}", "نعم", ar_indic(a["phone4"])]),
        status_call("ar02", "ar", "digits_as_words", b,
                    [f"{wb[0]} و{wb[1]}", "أيوه صح", ar_words(b["phone4"])]),
        status_call("ar03", "ar", "correction", c,
                    [f"رقم الطلب {misheard(d(c))}", f"لا، الرقم {d(c)}", "نعم", c["phone4"]]),
        status_call("ar04", "ar", "opener_without_number", e,
                    ["السلام عليكم، وين طلبي؟", f"الرقم {d(e)}", "إيه",
                     f"آخر أربعة أرقام {e['phone4']}"]),
        status_call("ar05", "ar", "full_phone_number", f,
                    [f"طلب رقم {d(f)}", "صحيح", f"رقمي ٠٥٠ ٠٠٠ {ar_indic(f['phone4'])}"]),
        handover_call("ar06", "ar", "handover_at_start", None, "requested",
                      ["أبي أكلم موظف لو سمحت"]),
        handover_call("ar07", "ar", "handover_mid_call", g["order_id"], "requested",
                      [f"رقم الطلب {d(g)}", "نعم", "ممكن أتكلم مع شخص من خدمة العملاء؟"]),
        handover_call("ar08", "ar", "wrong_phone_twice", h["order_id"], "repeated_failure",
                      [f"طلبي رقم {d(h)}", "نعم", plus(h["phone4"], 1111), f"الرقم {d(h)}",
                       "نعم", plus(h["phone4"], 2222)]),
        status_call("ar09", "ar", "wrong_phone_then_right", i,
                    [f"طلبي رقم {d(i)}", "نعم", plus(i["phone4"], 1111), f"آسف، {d(i)}",
                     "نعم", i["phone4"]]),
        handover_call("ar10", "ar", "unknown_order", "LS-" + UNKNOWN_ORDER, "repeated_failure",
                      [f"رقم الطلب {UNKNOWN_ORDER}", "نعم", "4821",
                       f"قلت لك {UNKNOWN_ORDER}", "نعم", "4821"]),
        status_call("ar11", "ar", "number_as_quantity", k,
                    ["رقم الطلب عشرة آلاف وخمسة وأربعين", ar_words(d(k)), "أيوه صح",
                     f"آخر أربعة أرقام {k['phone4']}"]),
        status_call("ar12", "ar", "fast_and_noisy", m,
                    [f"وين الطلب {d(m)} لو سمحت", "إيه إيه", m["phone4"]],
                    tags=("fast", "noisy")),
        status_call("ar13", "ar", "goodbye_after_status", n,
                    [f"رقم الطلب {ar_indic(d(n))}", "صح", n["phone4"],
                     "لا شكراً، مع السلامة"]),
        status_call("ar14", "ar", "missing_digit", p,
                    [f"رقم طلبي {d(p)[:4]}", f"آسف، {ar_words(d(p))}", "نعم", p["phone4"]]),
        status_call("ar15", "ar", "asks_to_repeat", q,
                    [f"طلبي رقم {d(q)}", "ممكن تعيد؟", "نعم", q["phone4"]]),
    ]


def build() -> list[dict]:
    by_status, by_id = load_orders()
    return english_calls(by_status, by_id) + arabic_calls(by_status, by_id)


def main() -> None:
    calls = build()
    used = [c["expected_order_id"] for c in calls if c["expected_outcome"] == "status"
            or c["category"] in ("handover_mid_call", "wrong_phone_twice")]
    assert len(set(used)) == len(used), "each call should use a different order"
    assert calls[1]["turns"][0].startswith("one double zero"), "en02 needs an LS-100xx order"
    with OUT.open("w", encoding="utf-8") as f:
        for call in calls:
            f.write(json.dumps(call, ensure_ascii=False) + "\n")
    print(f"Wrote {len(calls)} calls to {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
