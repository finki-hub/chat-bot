from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.schemas.chat_persistence import ActiveStreamStatus, JsonValue


class UserScopedRequest(BaseModel):
    user_id: UUID


class SetActiveStreamRequest(UserScopedRequest):
    active_stream_id: UUID
    active_response_id: UUID
    active_replacement_message_id: UUID | None = None
    active_status: ActiveStreamStatus


class ConversationUpdateRequest(UserScopedRequest):
    model: str | None = Field(default=None)
    title: str | None = Field(default=None)

    @field_validator("model", "title")
    @classmethod
    def _strip_present_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank or whitespace-only")
        return stripped


class UserMessageUpsertRequest(UserScopedRequest):
    id: UUID
    content: str = Field(min_length=1)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    parts: list[JsonValue] | None = None

    @field_validator("content")
    @classmethod
    def _strip_content(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank or whitespace-only")
        return stripped


class AssistantMessageUpsertRequest(UserMessageUpsertRequest):
    pass


class AssistantMessageReplacementRequest(UserScopedRequest):
    content: str = Field(min_length=1)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    parts: list[JsonValue] | None = None
    retained_message_ids: list[UUID] = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def _strip_content(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank or whitespace-only")
        return stripped


class ClearStaleActiveStreamsRequest(BaseModel):
    stale_before: datetime


class ClearStaleActiveStreamsResponse(BaseModel):
    cleared_count: int = Field(ge=0)
