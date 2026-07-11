import anyio
import pytest

from app.data.connection import Database
from app.llms import context as context_module
from app.llms.context import get_retrieved_context_with_sources
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
