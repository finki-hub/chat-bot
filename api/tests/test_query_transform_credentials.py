import anyio
import pytest

from app.data.connection import Database
from app.llms import context as context_module
from app.llms.context import _contextualize_query, get_retrieved_context_with_sources
from app.llms.models import Model
from app.llms.query_modes import QueryTransformMode
from app.llms.query_variants import QueryVariant, QueryVariantBundle


def test_retrieval_uses_raw_query_when_hosted_transform_credential_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_modes: list[QueryTransformMode] = []

    async def fail_if_contextualized(*args, **kwargs):
        raise AssertionError("contextualization must not call an unavailable provider")

    async def fake_build_query_variants(
        search_query,
        query_transform_model,
        mode,
        credentials,
    ):
        seen_modes.append(mode)
        raw = QueryVariant(kind="raw", text=search_query, is_document=False)
        return QueryVariantBundle(variants=(raw,), rerank_query=search_query)

    async def fake_embed_variant(*args, **kwargs):
        return [0.1]

    async def fake_search_both(*args, **kwargs):
        return [], []

    monkeypatch.setattr(
        context_module,
        "_contextualize_query",
        fail_if_contextualized,
    )
    monkeypatch.setattr(
        context_module,
        "build_query_variants",
        fake_build_query_variants,
    )
    monkeypatch.setattr(context_module, "_embed_variant", fake_embed_variant)
    monkeypatch.setattr(context_module, "_search_both", fake_search_both)

    async def collect():
        return await get_retrieved_context_with_sources(
            Database("postgresql://unused"),
            "original query",
            Model.BGE_M3_LOCAL,
            Model.GPT_5_4_MINI,
            query_transform_mode=QueryTransformMode.REWRITE_HYDE,
            history_text="private history",
        )

    result = anyio.run(collect)

    assert result.text == ""
    assert seen_modes == [QueryTransformMode.RAW]


def test_contextualization_uses_latest_query_when_provider_returns_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latest_query = "Колку чини тоа?"

    async def return_input(query, *args, **kwargs):
        return query

    monkeypatch.setattr(context_module, "transform_query", return_input)

    result = anyio.run(
        _contextualize_query,
        latest_query,
        Model.GPT_5_4_MINI,
        "Корисник: Колку чини пријавувањето?",
    )

    assert result == latest_query


def test_retrieval_uses_raw_depth_when_transformed_variants_are_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_limits: list[int] = []

    async def return_query(query, *args, **kwargs):
        return query

    async def return_raw_bundle(search_query, *args, **kwargs):
        raw = QueryVariant(kind="raw", text=search_query, is_document=False)
        return QueryVariantBundle(variants=(raw,), rerank_query=search_query)

    async def fake_embed_variant(*args, **kwargs):
        return [0.1]

    async def capture_search_limit(db, embedding, embedding_model, limit):
        search_limits.append(limit)
        return [], []

    monkeypatch.setattr(context_module, "has_provider_credential", lambda *args: True)
    monkeypatch.setattr(context_module, "_contextualize_query", return_query)
    monkeypatch.setattr(context_module, "build_query_variants", return_raw_bundle)
    monkeypatch.setattr(context_module, "_embed_variant", fake_embed_variant)
    monkeypatch.setattr(context_module, "_search_both", capture_search_limit)

    async def collect():
        return await get_retrieved_context_with_sources(
            Database("postgresql://unused"),
            "original query",
            Model.BGE_M3_LOCAL,
            Model.GPT_5_4_MINI,
            query_transform_mode=QueryTransformMode.REWRITE_HYDE,
        )

    result = anyio.run(collect)

    assert result.text == ""
    assert search_limits == [31]
