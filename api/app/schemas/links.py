from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl


class LinkSchema(BaseModel):
    id: UUID = Field(
        description="Unique identifier for the link",
        examples=["3fa85f64-5717-4562-b3fc-2c963f66afa6"],
    )
    name: str = Field(
        description="Key or title of the link",
        examples=["FINKI"],
    )
    url: HttpUrl = Field(
        description="Destination URL",
        examples=["https://finki.ukim.mk"],
    )
    description: str | None = Field(
        description="Optional human-readable description",
        examples=["FINKI homepage"],
    )
    user_id: str | None = Field(
        description="Discord ID of the user who created this link",
        examples=["198249751001563136"],
    )
    created_at: datetime = Field(
        description="Timestamp when the link was created (UTC)",
        examples=["2025-06-05T14:48:00Z"],
    )
    updated_at: datetime = Field(
        description="Timestamp when the link was last updated (UTC)",
        examples=["2025-06-06T09:24:00Z"],
    )


class CreateLinkSchema(BaseModel):
    name: str = Field(
        description="Key or title for the new link",
        examples=["FINKI"],
    )
    url: HttpUrl = Field(
        description="Destination URL for the link",
        examples=["https://finki.ukim.mk"],
    )
    description: str | None = Field(
        description="Optional description of the link",
        examples=["FINKI homepage"],
    )
    user_id: str | None = Field(
        description="Discord ID of the user creating the link",
        examples=["198249751001563136"],
    )


class UpdateLinkSchema(BaseModel):
    name: str | None = Field(
        default=None,
        description="New key/title for the link",
        examples=["example-updated"],
    )
    url: HttpUrl | None = Field(
        default=None,
        description="New destination URL",
        examples=["https://example.org"],
    )
    description: str | None = Field(
        default=None,
        description="Updated description",
        examples=["An updated example site"],
    )
    user_id: str | None = Field(
        default=None,
        description="Updated user ID for the link",
        examples=["user_456"],
    )
