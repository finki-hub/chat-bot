from dataclasses import dataclass
from datetime import UTC, datetime

from app.llms.model_catalog_types import ModelDescriptor
from app.llms.provider_credentials import ProviderName
from app.schemas.sponsored_access import (
    ModelAvailability,
    SponsoredAccessValidationError,
    SponsoredQuotaSnapshot,
)


@dataclass(frozen=True, slots=True)
class SponsoredSettings:
    enabled: bool
    provider_configured: bool = True


@dataclass(frozen=True, slots=True)
class ModelAccessContext:
    """User-specific access inputs kept outside the globally cached catalog."""

    available_providers: frozenset[ProviderName]
    sponsored_settings: SponsoredSettings
    personal_quota: SponsoredQuotaSnapshot | None
    global_quota: SponsoredQuotaSnapshot | None
    utc_reset: datetime
    sponsored_model_id: str | None = None
    sponsored_provider: ProviderName | None = None
    rejected_providers: frozenset[ProviderName] = frozenset()


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise SponsoredAccessValidationError(reason="utc_reset must be timezone-aware")
    return value.astimezone(UTC)


def _sponsored_configured(context: ModelAccessContext) -> bool:
    settings = context.sponsored_settings
    return settings.enabled and settings.provider_configured


def _sponsored_has_capacity(context: ModelAccessContext) -> bool:
    personal = context.personal_quota
    global_quota = context.global_quota
    return (
        _sponsored_configured(context)
        and context.sponsored_provider not in context.rejected_providers
        and personal is not None
        and global_quota is not None
        and personal.remaining > 0
        and global_quota.remaining > 0
    )


def _display_quota(context: ModelAccessContext) -> SponsoredQuotaSnapshot | None:
    if (
        not _sponsored_configured(context)
        or context.sponsored_provider in context.rejected_providers
        or context.personal_quota is None
    ):
        return None
    return context.personal_quota.model_copy(
        update={"resets_at": _utc(context.utc_reset)},
    )


def _sponsored_availability(context: ModelAccessContext) -> ModelAvailability:
    has_byok = context.sponsored_provider in context.available_providers
    has_sponsored = _sponsored_has_capacity(context)
    if has_byok and has_sponsored:
        return "both"
    if has_byok:
        return "byok"
    if has_sponsored:
        return "sponsored"
    return "unavailable"


def overlay_model_access(
    descriptor: ModelDescriptor,
    context: ModelAccessContext,
) -> ModelDescriptor:
    if (
        descriptor.id != context.sponsored_model_id
        or descriptor.provider != context.sponsored_provider
    ):
        availability: ModelAvailability = (
            "byok"
            if descriptor.provider in context.available_providers
            else "unavailable"
        )
        return descriptor.model_copy(
            update={"availability": availability, "sponsored_quota": None},
        )

    return descriptor.model_copy(
        update={
            "availability": _sponsored_availability(context),
            "sponsored_quota": _display_quota(context),
        },
    )
