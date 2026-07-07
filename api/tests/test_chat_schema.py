import pytest

from app.schemas.chat import ChatSchema


def test_chat_schema_rejects_more_than_10_messages():
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
        for i in range(11)
    ]
    messages[-1] = {"role": "user", "content": "last"}

    with pytest.raises(ValueError, match="List should have at most 10 items"):
        ChatSchema(messages=messages)


def test_chat_schema_rejects_turns_over_2000_chars():
    with pytest.raises(ValueError, match="String should have at most 2000 characters"):
        ChatSchema(messages=[{"role": "user", "content": "x" * 2001}])
