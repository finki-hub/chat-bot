import pytest

from app.llms.model_catalog_policy import MODEL_CATALOG
from app.llms.models import (
    ACTIVE_EMBEDDING_MODELS,
    LEGACY_CHAT_MODEL_ALIASES,
    Model,
    parse_chat_model,
)
from app.llms.provider_credentials import provider_for_model
from app.schemas.chat_title import ChatTitleSchema
from app.utils.settings import Settings


@pytest.mark.parametrize(
    ("legacy_id", "canonical"),
    [(alias, model) for alias, model in LEGACY_CHAT_MODEL_ALIASES.items()],
)
def test_title_schema_normalizes_legacy_model_ids(
    legacy_id: str,
    canonical: Model,
) -> None:
    payload = ChatTitleSchema.model_validate(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "query_transform_model": legacy_id,
            "provider_model": legacy_id,
        },
    )

    assert payload.query_transform_model is canonical
    assert payload.provider_model is canonical
    assert provider_for_model(payload.provider_model) == provider_for_model(canonical)


def test_legacy_openrouter_alias_is_not_accepted_as_dynamic_ollama() -> None:
    with pytest.raises(ValueError, match="active chat model"):
        parse_chat_model("openrouter:x-ai/unknown-model", frozenset())


def test_sponsored_legacy_model_setting_normalizes_to_current_id() -> None:
    settings = Settings(SPONSORED_MODEL_ID="gpt-5.6-luna")

    assert settings.SPONSORED_MODEL_ID == "gpt-6-luna"


def test_active_catalog_replaces_legacy_ids_and_preserves_embeddings() -> None:
    catalog_ids = {policy.model.value for policy in MODEL_CATALOG}

    assert set(LEGACY_CHAT_MODEL_ALIASES).isdisjoint(catalog_ids)
    assert {model.value for model in ACTIVE_EMBEDDING_MODELS} == {
        "BAAI/bge-m3",
        "gemini-embedding-001",
        "text-embedding-3-large",
    }
