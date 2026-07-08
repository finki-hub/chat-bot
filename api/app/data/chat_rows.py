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


def conversation_from_row(row: Mapping[str, object]) -> ChatConversation:
    return ChatConversation(
        id=row["id"],
        user_id=row["user_id"],
        active_stream_id=row["active_stream_id"],
        active_response_id=row["active_response_id"],
        active_status=row["active_status"],
        model=row["model"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def message_from_row(row: Mapping[str, object]) -> ChatMessage:
    return ChatMessage(
        id=row["id"],
        conversation_id=row["conversation_id"],
        role=row["role"],
        content=row["content"],
        response_id=row["response_id"],
        metadata=metadata_from_row(row["metadata"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
