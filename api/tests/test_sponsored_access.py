import json
from datetime import UTC, datetime

import pytest

from app.llms.agents import sponsored_error_event
from app.llms.model_access import (
    ModelAccessContext,
    SponsoredSettings,
    overlay_model_access,
)
from app.llms.model_catalog_types import (
    CatalogProvider,
    ExecutionPolicy,
    ModelDescriptor,
)
from app.llms.provider_credentials import ProviderName
from app.schemas.sponsored_access import (
    SPONSORED_ERROR_CODES,
    SafeErrorDetails,
    SponsoredQuotaSnapshot,
)

RESET = datetime(2026, 7, 19, tzinfo=UTC)


def _descriptor(
    model_id: str = "gpt-5.6-luna",
    provider: CatalogProvider = "openai",
) -> ModelDescriptor:
    return ModelDescriptor(
        id=model_id,
        provider=provider,
        name=model_id,
        execution=ExecutionPolicy(
            reasoning=True,
            sampling=True,
            tool_call=True,
            structured_output=True,
        ),
    )


def _context(
    *,
    available_providers: frozenset[ProviderName] = frozenset(),
    sponsored_enabled: bool = True,
    provider_configured: bool = True,
    personal_remaining: int = 4,
    global_remaining: int = 10,
    sponsored_model_id: str | None = "gpt-5.6-luna",
    sponsored_provider: ProviderName | None = "openai",
) -> ModelAccessContext:
    return ModelAccessContext(
        available_providers=available_providers,
        sponsored_model_id=sponsored_model_id,
        sponsored_provider=sponsored_provider,
        sponsored_settings=SponsoredSettings(
            enabled=sponsored_enabled,
            provider_configured=provider_configured,
        ),
        personal_quota=SponsoredQuotaSnapshot(
            limit=5,
            remaining=personal_remaining,
            resets_at=RESET,
        ),
        global_quota=SponsoredQuotaSnapshot(
            limit=100,
            remaining=global_remaining,
            resets_at=RESET,
        ),
        utc_reset=RESET,
    )


def test_legacy_model_descriptor_payload_parses_without_access_fields() -> None:
    payload = {
        "id": "gpt-5.6-sol",
        "provider": "openai",
        "name": "GPT-5.6 Sol",
        "execution": {
            "reasoning": True,
            "sampling": True,
            "tool_call": True,
            "structured_output": True,
        },
    }

    descriptor = ModelDescriptor.model_validate(payload)

    assert descriptor.availability == "byok"
    assert descriptor.sponsored_quota is None


@pytest.mark.parametrize(
    ("context", "expected"),
    [
        (_context(available_providers=frozenset({"openai"})), "both"),
        (
            _context(
                available_providers=frozenset({"openai"}),
                sponsored_enabled=False,
            ),
            "byok",
        ),
        (_context(), "sponsored"),
        (
            _context(global_remaining=0),
            "unavailable",
        ),
    ],
)
def test_sponsored_overlay_covers_all_availability_states(
    context: ModelAccessContext,
    expected: str,
) -> None:
    resolved = overlay_model_access(_descriptor(), context)

    assert resolved.availability == expected


def test_non_luna_overlay_never_grants_sponsored_access() -> None:
    resolved = overlay_model_access(
        _descriptor("gpt-5.6-sol"),
        _context(),
    )

    assert resolved.availability == "unavailable"
    assert resolved.sponsored_quota is None


@pytest.mark.parametrize(
    ("provider", "available_providers", "expected"),
    [
        ("openai", frozenset({"openai"}), "byok"),
        ("google", frozenset({"google"}), "byok"),
        ("anthropic", frozenset({"anthropic"}), "byok"),
        ("ollama", frozenset({"ollama"}), "byok"),
        ("ollama", frozenset({"openai"}), "unavailable"),
    ],
)
def test_non_luna_overlay_uses_descriptor_provider(
    provider: CatalogProvider,
    available_providers: frozenset[ProviderName],
    expected: str,
) -> None:
    resolved = overlay_model_access(
        _descriptor("provider-model", provider),
        _context(available_providers=available_providers),
    )

    assert resolved.availability == expected
    assert resolved.sponsored_quota is None


def test_overlay_is_user_specific_without_mutating_global_descriptor() -> None:
    base = _descriptor()

    byok_user = overlay_model_access(
        base,
        _context(available_providers=frozenset({"openai"})),
    )
    sponsored_user = overlay_model_access(base, _context())

    assert base.availability == "byok"
    assert base.sponsored_quota is None
    assert byok_user.availability == "both"
    assert sponsored_user.availability == "sponsored"
    assert byok_user.sponsored_quota is not None
    assert byok_user.sponsored_quota.resets_at == RESET


def test_safe_error_details_drop_unapproved_metadata() -> None:
    details = SafeErrorDetails.model_validate(
        {"resets_at": RESET, "global_limit": 100, "provider_error": "secret"},
    )

    assert details.model_dump(exclude_none=True) == {"resets_at": RESET}


def test_sponsored_error_event_uses_error_event_shape_and_safe_reset() -> None:
    exhausted = sponsored_error_event(
        "free_quota_exhausted",
        "Sponsored quota exhausted.",
        resets_at=RESET,
    )
    unavailable = sponsored_error_event(
        "free_tier_unavailable",
        "Sponsored access is unavailable.",
        resets_at=RESET,
    )
    in_progress = sponsored_error_event(
        "sponsored_request_in_progress",
        "Sponsored request already in progress.",
        resets_at=RESET,
    )

    exhausted_payload = json.loads(exhausted.split("data: ", 1)[1])
    unavailable_payload = json.loads(unavailable.split("data: ", 1)[1])
    assert exhausted_payload == {
        "code": "free_quota_exhausted",
        "message": "Sponsored quota exhausted.",
        "resets_at": "2026-07-19T00:00:00Z",
    }
    assert unavailable_payload == {
        "code": "free_tier_unavailable",
        "message": "Sponsored access is unavailable.",
    }
    assert json.loads(in_progress.split("data: ", 1)[1]) == {
        "code": "sponsored_request_in_progress",
        "message": "Sponsored request already in progress.",
    }
    assert SPONSORED_ERROR_CODES == (
        "free_quota_exhausted",
        "free_tier_unavailable",
        "sponsored_request_in_progress",
    )
