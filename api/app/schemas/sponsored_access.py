from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ModelAvailability = Literal["byok", "sponsored", "both", "unavailable"]
SponsoredErrorCode = Literal[
    "free_quota_exhausted",
    "free_tier_unavailable",
    "sponsored_request_in_progress",
]
SponsoredAdmissionOutcome = Literal[
    "admitted",
    "free_quota_exhausted",
    "free_tier_unavailable",
    "sponsored_request_in_progress",
]

SPONSORED_ERROR_CODES: Final[tuple[SponsoredErrorCode, ...]] = (
    "free_quota_exhausted",
    "free_tier_unavailable",
    "sponsored_request_in_progress",
)


@dataclass(frozen=True, slots=True)
class SponsoredAccessValidationError(ValueError):
    reason: str

    def __str__(self) -> str:
        return self.reason


class SponsoredQuotaSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    limit: int = Field(ge=0)
    remaining: int = Field(ge=0)
    resets_at: datetime

    @field_validator("resets_at")
    @classmethod
    def _normalize_reset_to_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise SponsoredAccessValidationError(
                reason="resets_at must be timezone-aware",
            )
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _remaining_cannot_exceed_limit(self) -> Self:
        if self.remaining > self.limit:
            raise SponsoredAccessValidationError(reason="remaining cannot exceed limit")
        return self

    def sse_reset(self) -> str:
        """Return the reset timestamp in the canonical SSE representation."""
        return self.resets_at.isoformat().replace("+00:00", "Z")


class SafeErrorDetails(BaseModel):
    """Client-safe sponsored error metadata; deployment details are excluded."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    resets_at: datetime | None = None

    @field_validator("resets_at")
    @classmethod
    def _normalize_reset_to_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise SponsoredAccessValidationError(
                reason="resets_at must be timezone-aware",
            )
        return value.astimezone(UTC)

    def sse_reset(self) -> str | None:
        """Return the approved reset field, or no metadata when no reset exists."""
        if self.resets_at is None:
            return None
        return self.resets_at.isoformat().replace("+00:00", "Z")


class SponsoredAdmissionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    outcome: SponsoredAdmissionOutcome
    quota: SponsoredQuotaSnapshot | None = None
    details: SafeErrorDetails | None = None


Availability = ModelAvailability
SponsoredSseCode = SponsoredErrorCode
SponsoredErrorDetails = SafeErrorDetails
