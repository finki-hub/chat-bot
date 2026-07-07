from app.schemas.chat import ChatSchema


def test_chat_schema_rejects_more_than_10_messages():
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
        for i in range(11)
    ]
    messages[-1] = {"role": "user", "content": "last"}

    try:
        ChatSchema(messages=messages)
    except ValueError:
        return

    raise AssertionError("ChatSchema accepted more than 10 messages")


def test_chat_schema_rejects_turns_over_2000_chars():
    try:
        ChatSchema(messages=[{"role": "user", "content": "x" * 2001}])
    except ValueError:
        return

    raise AssertionError("ChatSchema accepted a turn over 2000 characters")
