import pytest

from voice_agent.parser import detect_language, extract_digit_groups, parse


@pytest.mark.parametrize(
    "text, expected",
    [
        ("My order is LS-10045.", ["10045"]),
        ("1 0 0 4 5", ["10045"]),
        ("1-0-0-4-5", ["10045"]),
        ("10,045", ["10045"]),  # thousands separator from speech-to-text
        ("one zero zero four five", ["10045"]),
        ("one oh oh four five", ["10045"]),
        ("one double zero four five", ["10045"]),
        ("١٠٠٤٥", ["10045"]),  # Arabic-Indic digits
        ("واحد صفر صفر أربعة خمسة", ["10045"]),
        ("واحد صفر صفر أربعة وخمسة", ["10045"]),  # "and" attached to the last digit
        ("واحد صفر صفر أربعة و خمسة", ["10045"]),  # "and" as its own word
        ("The last four are 7669.", ["4", "7669"]),  # a word and a numeral never join
        ("order 10045 and phone 7669", ["10045", "7669"]),
        ("10045, 7669", ["10045", "7669"]),  # a comma joins only a 3-digit thousands group
        ("Oh, hello there", []),  # "oh" is only zero after another digit
        ("ten thousand and forty five", ["5"]),  # quantities are not parsed (by design)
    ],
)
def test_extract_digit_groups(text, expected):
    assert extract_digit_groups(text) == expected


def test_order_number_takes_the_last_five_digit_group():
    assert parse("not 10054, it's 10045").order_number == "10045"


@pytest.mark.parametrize(
    "text, last4",
    [
        ("7669", "7669"),
        ("seven six six nine", "7669"),
        ("My number is +971 50 000 7669.", "7669"),
        ("رقمي ٠٥٠ ٠٠٠ ٧٦٦٩", "7669"),
        ("10045", None),  # five digits is an order number, not a phone
        ("766", None),
    ],
)
def test_phone_last4(text, last4):
    assert parse(text).phone_last4 == last4


@pytest.mark.parametrize(
    "text, language",
    [
        ("Where is my order?", "en"),
        ("وين طلبي؟", "ar"),
        ("طلبي رقم LS-10045", "ar"),  # Arabic sentence with a Latin order prefix
        ("١٠٠٤٥", "ar"),
        ("10045", None),
    ],
)
def test_detect_language(text, language):
    assert detect_language(text) == language


@pytest.mark.parametrize(
    "text, intent",
    [
        ("I want to speak to a human, please.", "handover"),
        ("Can I talk to a real person?", "handover"),
        ("أبي أكلم موظف لو سمحت", "handover"),
        ("ممكن أتكلم مع شخص من خدمة العملاء؟", "handover"),
        ("Yes, that's right.", "yes"),
        ("أيوه صح", "yes"),
        ("إيه", "yes"),
        ("No, the order is 10054.", "no"),
        ("لا، الرقم ١٠٠٥٤", "no"),
        ("No, that's all. Thank you!", "goodbye"),
        ("لا شكراً، مع السلامة", "goodbye"),
        ("Sorry, can you repeat that?", "repeat"),
        ("ممكن تعيد؟", "repeat"),
        ("Hello, I'm calling to check where my parcel is.", "order_status"),
        ("السلام عليكم، وين طلبي؟", "order_status"),
    ],
)
def test_intents(text, intent):
    assert intent in parse(text).intents


def test_arabic_greeting_is_not_goodbye():
    assert "goodbye" not in parse("السلام عليكم").intents


# Transcripts returned by openai/gpt-4o-transcribe in the first live audio run
# (evals/results/audio_openai_gpt-4o-transcribe_2026-10-08_before-fixes_turns.jsonl).
@pytest.mark.parametrize(
    "transcript, order, last4",
    [
        ("Hi, my order number is LS10154.", "10154", None),  # STT dropped the hyphen
        ("ثمانية، ثلاثة، خمسة، صفر.", None, "8350"),  # STT added Arabic commas
        ("آخر أربع أرقام خمسة ثلاثة أربعة خمسة.", None, "5345"),
        ("Sorry, one zero zero zero seven.", "10007", None),
        ("رقم الطلب ١٠٩٩٩", "10999", None),
        ("אלס-10999.", "10999", None),  # "LS" came back in Hebrew letters
        ("12:41", None, "1241"),  # "twelve forty-one" written as a time
        ("10:35?", None, "1035"),
    ],
)
def test_real_stt_transcripts(transcript, order, last4):
    parsed = parse(transcript)
    assert parsed.order_number == order
    assert parsed.phone_last4 == last4


@pytest.mark.parametrize(
    "text, expected",
    [
        ("LS10045", ["10045"]),
        ("رقم10045", ["10045"]),  # an Arabic word glued to the number
        ("eight, two, six, zero", ["8260"]),  # commas between digit words join
        ("one, zero, zero, four, five", ["10045"]),
        ("10045, 7669", ["10045", "7669"]),  # but not between separate numerals
    ],
)
def test_letters_and_commas_around_digits(text, expected):
    assert extract_digit_groups(text) == expected


# Known gap, kept as a to-do: numbers said in pairs or as quantities. These are real
# transcripts from the live runs; the rules only read single digits. When a number-words
# parser (or an LLM fallback) is added, remove the xfail marks.
@pytest.mark.xfail(strict=True, reason="tens and quantities are not parsed yet")
@pytest.mark.parametrize(
    "transcript, digits",
    [
        ("سبعة وعشرين تسعين.", "2790"),  # "twenty-seven ninety", Arabic order: 7 and 20
        ("آسف، عشرة آلاف وعشرون", "10020"),  # "ten thousand and twenty"
        ("I said one oh nine ninety-nine.", "10999"),
    ],
)
def test_numbers_in_pairs_or_quantities(transcript, digits):
    assert digits in extract_digit_groups(transcript)
