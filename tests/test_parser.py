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
