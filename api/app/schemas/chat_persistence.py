from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

type JsonValue = (
    str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
)


class ActiveStreamStatus(StrEnum):
    PENDING = "pending"
    STREAMING = "streaming"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"


class ChatMessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class ChatConversationCreate(BaseModel):
    id: UUID = Field(description="Stable web conversation id minted by the BFF")
    user_id: str = Field(
        min_length=1,
        description="Anonymous web user id that owns the conversation",
    )
    model: str | None = Field(
        default=None,
        description="Selected inference model, if known",
    )
    title: str | None = Field(default=None, description="Display title, if known")

    @field_validator("user_id", "model", "title")
    @classmethod
    def _strip_present_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank or whitespace-only")
        return stripped


class ChatConversationUpdate(BaseModel):
    model: str | None = Field(
        default=None,
        description="Selected inference model, if known",
    )
    title: str | None = Field(default=None, description="Display title, if known")
    active_status: ActiveStreamStatus | None = Field(default=None)

    @field_validator("model", "title")
    @classmethod
    def _strip_present_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank or whitespace-only")
        return stripped


class ChatConversation(BaseModel):
    id: UUID
    user_id: str
    active_stream_id: UUID | None = None
    active_response_id: UUID | None = None
    active_status: ActiveStreamStatus | None = None
    model: str | None = None
    title: str | None = None
    created_at: datetime
    updated_at: datetime


class ChatMessageUpsert(BaseModel):
    id: UUID
    conversation_id: UUID
    role: ChatMessageRole
    content: str = Field(min_length=1)
    response_id: UUID | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def _strip_content(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank or whitespace-only")
        return stripped


class ChatMessage(BaseModel):
    id: UUID
    conversation_id: UUID
    role: ChatMessageRole
    content: str
    response_id: UUID | None = None
    metadata: dict[str, JsonValue]
    created_at: datetime
    updated_at: datetime


class ChatConversationWithMessages(BaseModel):
    conversation: ChatConversation
    messages: list[ChatMessage]
