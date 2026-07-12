import json
from collections.abc import Mapping, Sequence

from app.schemas.chat_persistence import (
    ChatConversation,
    ChatMessage,
    JsonValue,
)


def json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    return str(value)


def metadata_from_row(value: object) -> dict[str, JsonValue]:
    parsed: object = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, Mapping):
        return {}
    return {str(key): json_value(item) for key, item in parsed.items()}


def parts_from_row(value: object) -> list[JsonValue] | None:
    parsed: object = json.loads(value) if isinstance(value, str) else value
    if (
        parsed is None
        or not isinstance(parsed, Sequence)
        or isinstance(
            parsed,
            str | bytes | bytearray,
        )
    ):
        return None
    return [json_value(item) for item in parsed]


def conversation_from_row(row: Mapping[str, object]) -> ChatConversation:
    return ChatConversation.model_validate(dict(row))


def message_from_row(row: Mapping[str, object]) -> ChatMessage:
    values = dict(row)
    values["metadata"] = metadata_from_row(row["metadata"])
    values["parts"] = parts_from_row(row.get("parts"))
    return ChatMessage.model_validate(values)
