from datetime import datetime
from ipaddress import ip_address
from typing import Literal
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

ChatCredentialProvider = Literal["openai", "google", "anthropic"]


class ChatCredentialUpsert(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: ChatCredentialProvider
    api_key: str = Field(min_length=1)
    base_url: str | None = None

    @field_validator("api_key", "base_url")
    @classmethod
    def _strip_present_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank or whitespace-only")
        return stripped

    @field_validator("base_url")
    @classmethod
    def _base_url_must_be_public_https(cls, value: str | None) -> str | None:
        if value is None:
            return None

        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("base_url must be an https URL with a host")
        if parsed.username or parsed.password or parsed.fragment:
            raise ValueError("base_url must not include credentials or fragments")

        hostname = parsed.hostname.lower().rstrip(".")
        if hostname == "localhost" or hostname.endswith(".localhost"):
            raise ValueError("base_url must not target localhost")
        try:
            address = ip_address(hostname)
        except ValueError:
            if hostname.endswith(".local"):
                raise ValueError("base_url must not target local hostnames") from None
            return value

        if not address.is_global:
            raise ValueError("base_url must target a public host")
        return value


class ChatCredentialPublic(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: UUID
    provider: ChatCredentialProvider
    has_api_key: bool = True
    base_url: str | None = None
    created_at: datetime
    updated_at: datetime


class ChatCredentialSecret(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: ChatCredentialProvider
    api_key: str
    base_url: str | None = None
