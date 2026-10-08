"""Deterministic intent and slot parser for Arabic and English caller utterances.

No model is used here: order numbers and phone digits must be exact, so we parse them
with rules we can test. Input is the text from speech-to-text (or typed text).
"""

import re
from dataclasses import dataclass, field

ORDER_DIGITS = 5  # Lumi Skin order IDs look like LS-10045
PHONE_DIGITS = 4  # we verify with the last 4 digits of the phone on the order

# Arabic-Indic (٠-٩) and Eastern Arabic-Indic (۰-۹) digits -> 0-9
_DIGIT_TABLE = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_ARABIC_DIACRITICS = re.compile(r"[ً-ْـ]")  # short vowels + tatweel
_ARABIC_LETTER = re.compile(r"[ء-ي]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")
_ARABIC_INDIC_DIGIT = re.compile(r"[\u0660-\u0669\u06F0-\u06F9]")
_TOKEN = re.compile(r"\w+")
# A letter directly followed by a digit or the other way round: speech-to-text writes
# "LS-10154" as "LS10154", which would otherwise be one word with no number in it.
_LETTER_DIGIT_BOUNDARY = re.compile(r"(?<=[A-Za-zء-ي])(?=\d)|(?<=\d)(?=[A-Za-zء-ي])")

EN_DIGIT_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
EN_ZERO_AFTER_DIGIT = {"oh", "o"}  # "one oh oh four five"; only counted after a digit
EN_REPEATERS = {"double": 2, "triple": 3}  # "double zero" -> "00"

# Keys are already normalised (see normalise_arabic): no hamza on alef, final ه not ة.
AR_DIGIT_WORDS = {
    "صفر": "0",
    "واحد": "1",
    "اثنين": "2", "اثنان": "2", "اتنين": "2", "ثنين": "2",
    "ثلاثه": "3", "ثلاث": "3", "تلاته": "3",
    "اربعه": "4", "اربع": "4",
    "خمسه": "5", "خمس": "5",
    "سته": "6", "ست": "6",
    "سبعه": "7", "سبع": "7",
    "ثمانيه": "8", "ثماني": "8", "ثمان": "8", "تمانيه": "8",
    "تسعه": "9", "تسع": "9",
}
AR_JOINERS = {"و"}  # "اربعه و خمسه": a lone "and" between digit words keeps the group

# Intent keywords. Single words match whole tokens; phrases match inside the text.
INTENT_WORDS = {
    "handover": {
        "human", "agent", "person", "representative", "operator", "someone", "staff",
        "موظف", "شخص", "انسان", "ممثل", "بشري", "حد", "مسوول",
    },
    "yes": {
        "yes", "yeah", "yep", "yup", "correct", "right", "exactly", "sure", "affirmative",
        "نعم", "ايوه", "ايوا", "ايه", "اي", "صح", "صحيح", "اكيد", "تمام", "بالضبط", "مضبوط",
    },
    "no": {"no", "nope", "wrong", "incorrect", "لا", "غلط", "خطا", "مو", "مش", "ليس"},
    "goodbye": {"bye", "goodbye", "thanks", "شكرا", "باي", "خلاص"},
    "repeat": {"repeat", "pardon", "عيد", "تعيد", "كرر", "تكرر", "اعد", "اعيد"},
    "order_status": {
        "order", "parcel", "package", "delivery", "shipment", "track", "status", "where",
        "طلب", "طلبي", "طلبيه", "اوردر", "شحنه", "طرد", "توصيل", "وين",
    },
}
INTENT_PHRASES = {
    "handover": ["real person", "customer service", "خدمه العملاء"],
    "no": ["not right", "that's not"],
    "goodbye": ["thank you", "that's all", "that is all", "nothing else", "مع السلامه",
                "يعطيك العافيه"],
    "repeat": ["say that again", "come again", "what did you say", "didn't catch",
               "ما سمعت", "مره ثانيه"],
}


@dataclass
class Parsed:
    text: str
    language: str | None  # "ar", "en" or None when there are no letters to go on
    word_count: int = 0
    intents: set[str] = field(default_factory=set)
    digit_groups: list[str] = field(default_factory=list)
    order_number: str | None = None  # 5 digits, e.g. "10045"
    phone_last4: str | None = None


def normalise_arabic(text: str) -> str:
    """Remove short vowels and unify letter variants so keyword lists stay small."""
    text = _ARABIC_DIACRITICS.sub("", text)
    text = re.sub("[أإآ]", "ا", text)
    return text.replace("ة", "ه").replace("ى", "ي").replace("ؤ", "و")


def normalise(text: str) -> str:
    text = text.translate(_DIGIT_TABLE).replace("\u2019", "'")  # curly apostrophe from STT
    text = _LETTER_DIGIT_BOUNDARY.sub(" ", text)  # "LS10154" -> "LS 10154"
    return normalise_arabic(text).lower()


def detect_language(text: str) -> str | None:
    """'ar' or 'en' by majority script; Arabic-Indic digits count as Arabic.
    Returns None when there is nothing to go on (for example '10045')."""
    arabic = len(_ARABIC_LETTER.findall(text)) + len(_ARABIC_INDIC_DIGIT.findall(text))
    latin = len(_LATIN_LETTER.findall(text))
    if arabic == latin == 0:
        return None
    return "ar" if arabic >= latin else "en"


def count_words(text: str) -> int:
    """Words that are not numbers. Numbers say nothing about the caller's language: in a
    live run an Arabic caller's "9026" came back from STT as "nine zero two six", and
    counting those as four English words switched the call to English."""
    tokens = _TOKEN.findall(normalise(text))
    return sum(1 for i, token in enumerate(tokens)
               if not token.isdigit() and _token_digits(tokens, i) is None
               and token not in EN_REPEATERS)


def _arabic_variants(token: str) -> list[str]:
    """The token plus versions without a leading 'و' (and) or 'ال' (the)."""
    variants = [token]
    if token.startswith("و") and len(token) > 3:
        variants.append(token[1:])
    for prefix in ("ال", "بال", "لل"):
        for v in list(variants):
            if v.startswith(prefix) and len(v) - len(prefix) >= 3:
                variants.append(v[len(prefix):])
    return variants


def _token_digits(tokens: list[str], i: int) -> str | None:
    """Digits spoken by tokens[i] as a word ('five', 'خمسه', 'double'), else None."""
    token = tokens[i]
    if token in EN_DIGIT_WORDS:
        return EN_DIGIT_WORDS[token]
    if token in EN_ZERO_AFTER_DIGIT and i > 0 and _token_digits(tokens, i - 1) is not None:
        return "0"
    for variant in _arabic_variants(token):
        if variant in AR_DIGIT_WORDS:
            return AR_DIGIT_WORDS[variant]
    return None


def extract_digit_groups(text: str) -> list[str]:
    """Find spoken or written numbers and return them as digit strings.

    Rules (kept simple so they are easy to explain):
    - numerals next to numerals join when separated by spaces or hyphens
      ("1 0 0 4 5", "1-0-0-4-5"), or by a thousands comma/dot ("10,045");
    - digit words next to digit words join ("one zero zero four five"), also across a
      comma, because speech-to-text adds them ("ثمانية، ثلاثة، خمسة، صفر");
    - numerals around a colon join: STT writes "twelve forty-one" as the time "12:41";
    - a numeral and a digit word never join ("last four 7669" stays "4" and "7669");
    - "double"/"triple" repeat the next digit word; a lone Arabic "و" keeps a group going.
    """
    text = normalise(text)
    matches = list(_TOKEN.finditer(text))
    tokens = [m.group() for m in matches]
    groups: list[str] = []
    current, kind, prev_end, repeat = "", None, 0, 1

    for i, match in enumerate(matches):
        token = tokens[i]
        separator = text[prev_end:match.start()]
        prev_end = match.end()

        if token in EN_REPEATERS:
            repeat = EN_REPEATERS[token]
            continue
        if token in AR_JOINERS and kind == "word":
            continue

        if token.isdigit():
            token_kind, digits = "numeral", token
        else:
            word_digits = _token_digits(tokens, i)
            token_kind, digits = ("word", word_digits) if word_digits else (None, None)

        if token_kind is None:  # a normal word ends the current number
            if current:
                groups.append(current)
            current, kind, repeat = "", None, 1
            continue

        digits = digits * repeat
        repeat = 1
        if current and token_kind == kind and _joins(separator, kind, token):
            current += digits
        else:
            if current:
                groups.append(current)
            current, kind = digits, token_kind

    if current:
        groups.append(current)
    return groups


def _joins(separator: str, kind: str, token: str) -> bool:
    separator = separator.strip()
    if separator in ("", "-"):
        return True
    # "eight, two, six" or "ثمانية، ثلاثة": STT puts commas between spoken digits
    if kind == "word" and separator in (",", "،"):
        return True
    # "12:41": a phone ending said in pairs ("twelve forty-one") comes back as a time
    if kind == "numeral" and separator == ":":
        return True
    # "10,045" or "10.045": a thousands separator between numerals
    return kind == "numeral" and separator in (",", ".") and len(token) == 3


def find_order_number(groups: list[str]) -> str | None:
    """The last 5-digit group (the last one wins, so corrections work)."""
    matches = [g for g in groups if len(g) == ORDER_DIGITS]
    return matches[-1] if matches else None


def find_phone_last4(groups: list[str]) -> str | None:
    """Exactly 4 digits, or a full phone number (9+ digits) from which we take the last 4."""
    for group in reversed(groups):
        if len(group) == PHONE_DIGITS or len(group) >= 9:
            return group[-PHONE_DIGITS:]
    return None


def comparable_words(text: str) -> list[str]:
    """Normalised words with every number split into single digits, used to compare a
    transcript with the script it came from: '10154', 'one zero one five four' and
    '١٠١٥٤' all become ['1', '0', '1', '5', '4']. Punctuation and case are ignored."""
    tokens = _TOKEN.findall(normalise(text))
    words, repeat = [], 1
    for i, token in enumerate(tokens):
        if token in EN_REPEATERS:
            repeat = EN_REPEATERS[token]
            continue
        digits = token if token.isdigit() else _token_digits(tokens, i)
        words.extend(list(digits * repeat) if digits else [token])
        repeat = 1
    return words


def detect_intents(text: str) -> set[str]:
    norm = normalise(text)
    tokens = set()
    for token in _TOKEN.findall(norm):
        tokens.update(_arabic_variants(token))
    intents = {name for name, words in INTENT_WORDS.items() if tokens & words}
    for name, phrases in INTENT_PHRASES.items():
        if any(phrase in norm for phrase in phrases):
            intents.add(name)
    return intents


def parse(text: str) -> Parsed:
    groups = extract_digit_groups(text)
    return Parsed(
        text=text,
        language=detect_language(text),
        word_count=count_words(text),
        intents=detect_intents(text),
        digit_groups=groups,
        order_number=find_order_number(groups),
        phone_last4=find_phone_last4(groups),
    )
