from conftest import talk

from voice_agent.agent import ASK_ORDER, ASK_PHONE, CONFIRM_ORDER, DONE, ENDED, HANDOVER
from voice_agent.replies import GREETING, say


def test_greeting_discloses_automation_in_both_languages():
    assert "automated assistant" in GREETING and "مساعد آلي" in GREETING


def test_happy_path_english(agent, orders):
    order = orders["LS-10154"]
    t1, t2, t3 = talk(agent, "My order is LS-10154.", "Yes.", order["phone4"])
    assert t1.stage == CONFIRM_ORDER and "one, zero, one, five, four" in t1.reply
    assert t2.stage == ASK_PHONE
    assert t3.stage == DONE and "shipped" in t3.reply
    assert [c.name for c in t3.tool_calls] == ["get_order", "get_tracking"]
    assert agent.state.reported == [("LS-10154", order["status"])]


def test_happy_path_arabic_replies_in_arabic(agent, orders):
    order = orders["LS-10154"]
    *_, last = talk(agent, "مرحبا، رقم طلبي ١٠١٥٤", "نعم", order["phone4"])
    assert last.language == "ar" and "تم شحن طلبك" in last.reply


def test_no_tool_call_before_confirmation_and_phone(agent):
    turns = talk(agent, "10154", "yes")
    assert all(not t.tool_calls for t in turns)


def test_correction_is_read_back_again(agent):
    t1, t2 = talk(agent, "My order is 10128.", "No, the order is 10182.")
    assert t2.stage == CONFIRM_ORDER and "one, zero, one, eight, two" in t2.reply
    assert agent.state.failures == 0


def test_same_message_for_unknown_order_and_wrong_phone(tools, orders):
    from voice_agent.agent import OrderStatusAgent

    a, b = OrderStatusAgent(tools), OrderStatusAgent(tools)
    unknown = talk(a, "10999", "yes", "1234")[-1]
    wrong = talk(b, "10154", "yes", "0000")[-1]
    assert unknown.reply == wrong.reply == say("verify_failed", "en")


def test_two_failed_verifications_hand_over(agent, orders):
    *_, last = talk(agent, "10154", "yes", "0000", "10154", "yes", "1111")
    assert last.stage == HANDOVER and agent.state.handover_reason == "repeated_failure"
    assert last.tool_calls[-1].name == "handover_to_human"
    assert agent.state.reported == []  # nothing was revealed


def test_handover_on_request_at_any_stage(agent):
    talk(agent, "10154", "yes")
    last = agent.handle("Actually, can I talk to a real person?")
    assert last.stage == HANDOVER and agent.state.handover_reason == "requested"
    assert last.tool_calls[0].args["order_id"] == "10154"


def test_where_is_my_parcel_is_not_a_failure(agent):
    turn = agent.handle("Hello, where is my parcel?")
    assert turn.stage == ASK_ORDER and agent.state.failures == 0


def test_wrong_length_number_counts_as_a_failure(agent):
    turn = agent.handle("My order is 1004.")
    assert turn.stage == ASK_ORDER and agent.state.failures == 1
    assert "five digits" in turn.reply


def test_another_order_must_be_verified_again(agent, orders):
    talk(agent, "10154", "yes", orders["LS-10154"]["phone4"])
    turn = agent.handle("And order 10155?")
    assert turn.stage == CONFIRM_ORDER and not turn.tool_calls


def test_goodbye_ends_the_call(agent, orders):
    *_, last = talk(agent, "10154", "yes", orders["LS-10154"]["phone4"], "No, that's all.")
    assert last.stage == ENDED


def test_one_word_reply_does_not_switch_language(agent):
    agent.handle("رقم الطلب ١٠١٥٤")
    turn = agent.handle("OK yes")  # two words: a clear switch is allowed
    assert turn.language == "en"
    agent.handle("نعم")
    assert agent.state.language == "en"  # one word: keep the current language


def test_repeat_returns_the_last_reply(agent):
    first = agent.handle("10154")
    again = agent.handle("Sorry, can you repeat that?")
    assert again.reply == first.reply and agent.state.failures == 0


def test_digits_in_english_words_do_not_switch_an_arabic_call(agent):
    # Live run (scribe-v2): an Arabic caller's "9026" was transcribed as English words.
    agent.handle("طلبي رقم 10018")
    turn = agent.handle("nine zero two six")
    assert turn.language == "ar"
