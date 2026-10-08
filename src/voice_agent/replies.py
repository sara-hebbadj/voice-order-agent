"""Everything the agent says, in English and Arabic.

Replies are written for the ear: short sentences, digits read one by one, no personal
data (no name, address or amount), and dates spoken as "2 October".
"""

EN_DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
AR_DIGITS = ["صفر", "واحد", "اثنين", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة"]
EN_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
             "September", "October", "November", "December"]
AR_MONTHS = ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس",
             "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"]

# The greeting is bilingual because we do not know the caller's language yet.
# It also says the caller is talking to an automated assistant (AI transparency).
GREETING = (
    "Welcome to Lumi Skin. I'm an automated assistant, and you can speak English or Arabic. "
    "Please tell me your order number. "
    "أهلاً بك في لومي سكين. أنا مساعد آلي، وتقدر تتكلم عربي أو إنجليزي. من فضلك قل رقم الطلب."
)

TEMPLATES = {
    "ask_order": {
        "en": "Sure, I can check that. What is your order number?",
        "ar": "أكيد، أقدر أتحقق من ذلك. ما هو رقم الطلب؟",
    },
    "reprompt_order": {
        "en": "Sorry, I didn't catch an order number. Please say it digit by digit.",
        "ar": "عذراً، لم أسمع رقم الطلب. من فضلك قله رقماً رقماً.",
    },
    "reprompt_order_length": {
        "en": "Order numbers have five digits, like one, zero, zero, four, five. "
              "Please say yours digit by digit.",
        "ar": "رقم الطلب من خمسة أرقام، مثل واحد، صفر، صفر، أربعة، خمسة. "
              "من فضلك قله رقماً رقماً.",
    },
    "confirm_order": {
        "en": "I heard order number {digits}. Is that right?",
        "ar": "سمعت رقم الطلب {digits}. هل هذا صحيح؟",
    },
    "reprompt_confirm": {
        "en": "Please say yes or no. Is the order number {digits}?",
        "ar": "من فضلك قل نعم أو لا. هل رقم الطلب {digits}؟",
    },
    "ask_order_again": {
        "en": "Sorry about that. Please say the order number again, digit by digit.",
        "ar": "آسف على ذلك. من فضلك قل رقم الطلب مرة ثانية، رقماً رقماً.",
    },
    "ask_phone": {
        "en": "Thank you. For your security, please tell me the last four digits "
              "of the phone number on the order.",
        "ar": "شكراً. للتحقق من هويتك، من فضلك قل آخر أربعة أرقام من رقم الهاتف المسجل في الطلب.",
    },
    "reprompt_phone": {
        "en": "Sorry, I need the last four digits of the phone number on the order.",
        "ar": "عذراً، أحتاج آخر أربعة أرقام من رقم الهاتف المسجل في الطلب.",
    },
    # Same message whether the order does not exist or the phone does not match,
    # so a caller cannot use the bot to find out which order numbers exist.
    "verify_failed": {
        "en": "Sorry, I couldn't match those details. Let's try again. "
              "Please say your order number.",
        "ar": "عذراً، لم أستطع مطابقة هذه البيانات. لنحاول مرة ثانية. من فضلك قل رقم الطلب.",
    },
    "anything_else": {
        "en": "Is there anything else I can help with?",
        "ar": "هل أقدر أساعدك بشيء آخر؟",
    },
    "other_order": {
        "en": "Sure. What is the other order number?",
        "ar": "أكيد. ما هو رقم الطلب الآخر؟",
    },
    "only_order_status": {
        "en": "I can only help with order status on this line. Is there anything else?",
        "ar": "أقدر أساعد فقط في حالة الطلب على هذا الخط. هل هناك شيء آخر؟",
    },
    "goodbye": {
        "en": "Thank you for calling Lumi Skin. Goodbye.",
        "ar": "شكراً لاتصالك بلومي سكين. مع السلامة.",
    },
    "handover_requested": {
        "en": "Of course. I'm transferring you to a colleague now. "
              "I've passed on what you told me, so you won't need to repeat it.",
        "ar": "أكيد. سأحولك الآن إلى أحد زملائي، وأرسلت له ما قلته حتى لا تحتاج إلى تكراره.",
    },
    "handover_failure": {
        "en": "I'm sorry I couldn't sort this out. "
              "I'm transferring you to a colleague who can help.",
        "ar": "آسف لأنني لم أستطع المساعدة. سأحولك الآن إلى أحد زملائي ليساعدك.",
    },
}

STATUS_TEMPLATES = {
    "processing": {
        "en": "Your order was placed on {order_date} and is being prepared. "
              "It has not shipped yet.",
        "ar": "تم تسجيل طلبك في {order_date} ويجري تجهيزه الآن، ولم يتم شحنه بعد.",
    },
    "shipped": {
        "en": "Your order shipped on {shipped_date}. Latest update: {event}, {location}, "
              "on {event_date}. {delivery_window}",
        "ar": "تم شحن طلبك في {shipped_date}. آخر تحديث: {event}، {location}، في {event_date}. "
              "{delivery_window}",
    },
    "delivered": {
        "en": "Your order was delivered on {delivered_date}.",
        "ar": "تم توصيل طلبك في {delivered_date}.",
    },
    "returned": {
        "en": "Your order was returned to us after delivery on {delivered_date}. "
              "If you are waiting for a refund, a colleague can help.",
        "ar": "تم إرجاع طلبك إلينا بعد توصيله في {delivered_date}. "
              "إذا كنت تنتظر استرداد المبلغ، يمكن لأحد زملائي مساعدتك.",
    },
    "cancelled": {
        "en": "Your order was cancelled, so it will not be shipped.",
        "ar": "تم إلغاء طلبك، لذلك لن يتم شحنه.",
    },
}

EVENTS = {
    "picked_up": {"en": "picked up by the courier", "ar": "استلمه المندوب"},
    "in_transit": {"en": "in transit", "ar": "في الطريق"},
    "out_for_delivery": {"en": "out for delivery", "ar": "خرج للتوصيل"},
    "delivered": {"en": "delivered", "ar": "تم التوصيل"},
    "failed_attempt": {"en": "a delivery attempt failed", "ar": "محاولة توصيل لم تنجح"},
}

DELIVERY_WINDOW = {  # from the shop policy: UAE 1-3 business days, GCC 3-7
    "uae": {"en": "Delivery in the UAE usually takes 1 to 3 business days.",
            "ar": "التوصيل داخل الإمارات يستغرق عادة من يوم إلى ثلاثة أيام عمل."},
    "gcc": {"en": "Delivery in the GCC usually takes 3 to 7 business days.",
            "ar": "التوصيل لدول الخليج يستغرق عادة من ثلاثة إلى سبعة أيام عمل."},
}

AR_CITIES = {
    "Dubai": "دبي", "Abu Dhabi": "أبوظبي", "Sharjah": "الشارقة", "Ajman": "عجمان",
    "Ras Al Khaimah": "رأس الخيمة", "Al Ain": "العين", "Muscat": "مسقط", "Doha": "الدوحة",
    "Riyadh": "الرياض", "Jeddah": "جدة", "Kuwait City": "مدينة الكويت", "Kuwait": "الكويت",
    "Manama": "المنامة",
}


def say(key: str, language: str, **values) -> str:
    return TEMPLATES[key][language].format(**values)


def spell_digits(digits: str, language: str) -> str:
    """'10045' -> 'one, zero, zero, four, five' so TTS reads it digit by digit."""
    words = AR_DIGITS if language == "ar" else EN_DIGITS
    separator = "، " if language == "ar" else ", "
    return separator.join(words[int(d)] for d in digits)


def spoken_date(iso_date: str | None, language: str) -> str:
    """'2026-10-02' or '2026-10-02T14:00' -> '2 October'."""
    if not iso_date:
        return ""
    _, month, day = iso_date[:10].split("-")
    months = AR_MONTHS if language == "ar" else EN_MONTHS
    return f"{int(day)} {months[int(month) - 1]}"


def location_in_arabic(location: str) -> str:
    """Translate the courier locations in the data ('Dubai Hub', 'Riyadh Sorting Centre')."""
    if location.startswith("Lumi Skin Warehouse"):
        return "مستودع لومي سكين في دبي"
    for suffix, arabic in ((" Sorting Centre", "مركز الفرز في "), (" Hub", "مركز ")):
        if location.endswith(suffix):
            city = location.removesuffix(suffix)
            return arabic + AR_CITIES.get(city, city)
    return AR_CITIES.get(location, location)


def status_reply(order: dict, tracking: dict | None, language: str) -> str | None:
    """Turn the API's order + tracking JSON into one spoken sentence or two.
    Returns None for a status we have no template for (the agent then hands over)."""
    values = {
        "order_date": spoken_date(order.get("order_date"), language),
        "shipped_date": spoken_date(order.get("shipped_date"), language),
        "delivered_date": spoken_date(order.get("delivered_date"), language),
        "event": "", "location": "", "event_date": "", "delivery_window": "",
    }
    events = (tracking or {}).get("events") or []
    if events:
        last = events[-1]
        values["event"] = EVENTS.get(last["event"], {}).get(language, last["event"])
        location = last["location"]
        values["location"] = location_in_arabic(location) if language == "ar" else location
        values["event_date"] = spoken_date(last["timestamp"], language)
    region = "uae" if order.get("destination_country") == "United Arab Emirates" else "gcc"
    values["delivery_window"] = DELIVERY_WINDOW[region][language]

    template = STATUS_TEMPLATES.get(order["status"])
    if template is None:
        return None
    return template[language].format(**values).strip()
