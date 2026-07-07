from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.schemas.chat_persistence import ActiveStreamStatus, JsonValue


class UserScopedRequest(BaseModel):
    user_id: UUID


class SetActiveStreamRequest(UserScopedRequest):
    active_stream_id: UUID
    active_response_id: UUID
    active_status: ActiveStreamStatus


class UserMessageUpsertRequest(UserScopedRequest):
    id: UUID
    content: str = Field(min_length=1)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def _strip_content(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank or whitespace-only")
        return stripped


class AssistantMessageUpsertRequest(UserMessageUpsertRequest):
    pass


class ClearStaleActiveStreamsRequest(BaseModel):
    stale_before: datetime


class ClearStaleActiveStreamsResponse(BaseModel):
    cleared_count: int = Field(ge=0)
