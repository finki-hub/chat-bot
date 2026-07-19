import json
import secrets
from datetime import UTC, datetime

import pytest

from app.api.sponsored_access import resolve_sponsored_inference
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
from app.schemas.chat_credentials import ChatCredentialSecret
from app.schemas.sponsored_access import (
    SPONSORED_ERROR_CODES,
    SafeErrorDetails,
    SponsoredQuotaSnapshot,
)
from app.utils.settings import Settings

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


def test_sponsored_resolution_prefers_user_credential_even_when_user_key_is_rejected() -> (
    None
):
    user_credential = ChatCredentialSecret(
        provider="openai",
        api_key=secrets.token_urlsafe(),
        base_url="https://user.example/v1",
    )
    settings = Settings(
        SPONSORED_MODEL_ENABLED=True,
        SPONSORED_MODEL_API_KEY="sponsored-key",
        SPONSORED_MODEL_BASE_URL="HTTPS://Sponsored.EXAMPLE/v1/",
        SPONSORED_MODEL_UPSTREAM_MODEL="upstream-luna",
        SPONSORED_DAILY_GLOBAL_LIMIT=10,
    )

    resolved = resolve_sponsored_inference(
        "gpt-5.6-luna",
        user_credential,
        settings,
        user_credential_rejected=True,
    )

    assert resolved.credential == user_credential
    assert resolved.sponsored is False
    assert resolved.upstream_model is None


def test_sponsored_resolution_builds_inference_only_sponsored_credential() -> None:
    settings = Settings(
        SPONSORED_MODEL_ENABLED=True,
        SPONSORED_MODEL_API_KEY="sponsored-key",
        SPONSORED_MODEL_BASE_URL="HTTPS://Sponsored.EXAMPLE/v1/",
        SPONSORED_MODEL_UPSTREAM_MODEL="upstream-luna",
        SPONSORED_MAX_OUTPUT_TOKENS=1024,
        SPONSORED_DAILY_GLOBAL_LIMIT=10,
    )

    resolved = resolve_sponsored_inference("gpt-5.6-luna", None, settings)

    assert resolved.credential == ChatCredentialSecret(
        provider="openai",
        api_key="sponsored-key",
        base_url="https://sponsored.example/v1",
    )
    assert resolved.sponsored is True
    assert resolved.upstream_model == "upstream-luna"
    assert resolved.max_output_tokens == 1024


def test_sponsored_resolution_does_not_sponsor_a_rejected_user_credential() -> None:
    settings = Settings(
        SPONSORED_MODEL_ENABLED=True,
        SPONSORED_MODEL_API_KEY="sponsored-key",
        SPONSORED_DAILY_GLOBAL_LIMIT=10,
    )

    resolved = resolve_sponsored_inference(
        "gpt-5.6-luna",
        None,
        settings,
        user_credential_rejected=True,
    )

    assert resolved.credential is None
    assert resolved.sponsored is False


def test_sponsored_resolution_targets_only_configured_model_id() -> None:
    user_credential = ChatCredentialSecret(provider="google", api_key="user-key")
    settings = Settings(
        SPONSORED_MODEL_ENABLED=True,
        SPONSORED_MODEL_ID="gpt-5.6-luna",
        SPONSORED_MODEL_PROVIDER="openai",
        SPONSORED_MODEL_API_KEY="sponsored-key",
        SPONSORED_DAILY_GLOBAL_LIMIT=10,
    )

    resolved = resolve_sponsored_inference(
        "gemini-3.5-flash",
        user_credential,
        settings,
    )

    assert resolved.credential == user_credential
    assert resolved.sponsored is False
    assert resolved.upstream_model is None


def test_sponsored_resolution_uses_configured_provider_for_credential_secret() -> None:
    settings = Settings(
        SPONSORED_MODEL_ENABLED=True,
        SPONSORED_MODEL_ID="gemini-3.5-flash",
        SPONSORED_MODEL_PROVIDER="google",
        SPONSORED_MODEL_API_KEY="sponsored-key",
        SPONSORED_DAILY_GLOBAL_LIMIT=10,
    )

    resolved = resolve_sponsored_inference("gemini-3.5-flash", None, settings)

    assert resolved.credential == ChatCredentialSecret(
        provider="google",
        api_key="sponsored-key",
    )
    assert resolved.sponsored is True
