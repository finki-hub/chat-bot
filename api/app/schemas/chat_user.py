from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

ChatUserProvider = Literal["google", "microsoft-entra-id", "discord"]


class ChatUserUpsert(BaseModel):
    provider: ChatUserProvider
    provider_subject: str = Field(min_length=1)
    email: str | None = None
    name: str | None = None
    avatar_url: str | None = None

    @field_validator("provider", "provider_subject", "email", "name", "avatar_url")
    @classmethod
    def _strip_present_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank or whitespace-only")
        return stripped


class ChatUser(BaseModel):
    id: UUID
    provider: str
    provider_subject: str
    email: str | None = None
    name: str | None = None
    avatar_url: str | None = None
    created_at: datetime
    updated_at: datetime
