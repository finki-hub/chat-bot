import logging
from dataclasses import dataclass

import anyio
import pytest
from anyio.lowlevel import checkpoint

from app.data.connection import Database
from app.llms import anthropic as anthropic_module
from app.llms import context as context_module
from app.llms import google as google_module
from app.llms import ollama as ollama_module
from app.llms import openai as openai_module
from app.llms import query_transform as query_transform_module
from app.llms.context import _contextualize_query, get_retrieved_context_with_sources
from app.llms.models import Model
from app.llms.query_modes import QueryTransformMode
from app.llms.query_variants import QueryVariant, QueryVariantBundle


@dataclass(frozen=True, slots=True)
class _FakeResponse:
    content: str = "transformed query"


class _FakeLlm:
    async def ainvoke(self, messages):
        await checkpoint()
        return _FakeResponse()


def test_contextualize_query_logs_lengths_without_raw_text(caplog, monkeypatch):
    async def fake_transform_query(*args, **kwargs):
        await checkpoint()
        return "rewritten private query"

    monkeypatch.setattr(context_module, "transform_query", fake_transform_query)
    caplog.set_level(logging.INFO, logger="app.llms.context")

    result = anyio.run(
        _contextualize_query,
        "original private query",
        Model.CLAUDE_HAIKU_4_5,
        "private history",
    )

    assert result == "rewritten private query"
    assert "original private query" not in caplog.text
    assert "rewritten private query" not in caplog.text
    assert "private history" not in caplog.text
    assert "query_len=" in caplog.text
    assert "condensed_len=" in caplog.text
    assert "history_char_len=" in caplog.text


def test_retrieval_logs_lengths_without_raw_query(caplog, monkeypatch):
    async def fake_contextualize_query(*args, **kwargs):
        await checkpoint()
        return "private retrieval query"

    async def fake_build_query_variants(*args, **kwargs):
        await checkpoint()
        raw = QueryVariant(
            kind="raw",
            text="private retrieval query",
            is_document=False,
        )
        return QueryVariantBundle(
            variants=(raw,),
            rerank_query="private retrieval query",
        )

    async def fake_embed_variant(*args, **kwargs):
        await checkpoint()
        return [0.1]

    async def fake_search_both(*args, **kwargs):
        await checkpoint()
        return [], []

    monkeypatch.setattr(
        context_module,
        "_contextualize_query",
        fake_contextualize_query,
    )
    monkeypatch.setattr(
        context_module,
        "build_query_variants",
        fake_build_query_variants,
    )
    monkeypatch.setattr(context_module, "_embed_variant", fake_embed_variant)
    monkeypatch.setattr(context_module, "_search_both", fake_search_both)
    caplog.set_level(logging.INFO, logger="app.llms.context")

    async def collect():
        return await get_retrieved_context_with_sources(
            Database("postgresql://unused"),
            "private retrieval query",
            Model.BGE_M3,
            Model.CLAUDE_HAIKU_4_5,
            query_transform_mode=QueryTransformMode.RAW,
        )

    result = anyio.run(collect)

    assert result.text == ""
    assert "private retrieval query" not in caplog.text
    assert "query_len=" in caplog.text
    assert "embedding_model=" in caplog.text


def test_query_transform_logs_lengths_without_raw_query(caplog, monkeypatch):
    async def fake_transform_query_with_openai(*args, **kwargs):
        await checkpoint()
        return "transformed query"

    monkeypatch.setattr(
        query_transform_module,
        "transform_query_with_openai",
        fake_transform_query_with_openai,
    )
    caplog.set_level(logging.INFO, logger="app.llms.query_transform")

    async def collect():
        return await query_transform_module.transform_query(
            "private transform query",
            Model.GPT_5_4_MINI,
            temperature=0.0,
            top_p=1.0,
            max_tokens=128,
        )

    result = anyio.run(collect)

    assert result == "transformed query"
    assert "private transform query" not in caplog.text
    assert "query_length=" in caplog.text
    assert "model=" in caplog.text


@pytest.mark.parametrize(
    ("module", "function_name", "client_factory", "model"),
    [
        (
            openai_module,
            "transform_query_with_openai",
            "get_openai_llm",
            Model.GPT_5_4_MINI,
        ),
        (
            google_module,
            "transform_query_with_google",
            "get_google_llm",
            Model.GEMINI_3_5_FLASH,
        ),
        (
            anthropic_module,
            "transform_query_with_anthropic",
            "get_anthropic_llm",
            Model.CLAUDE_HAIKU_4_5,
        ),
        (ollama_module, "transform_query_with_ollama", "get_llm", Model.QWEN3_14B),
    ],
)
def test_provider_query_transform_logs_lengths_without_raw_query(
    caplog,
    monkeypatch,
    module,
    function_name,
    client_factory,
    model,
):
    monkeypatch.setattr(module, client_factory, lambda *args, **kwargs: _FakeLlm())
    caplog.set_level(logging.INFO, logger=module.__name__)

    async def collect():
        return await getattr(module, function_name)(
            "private provider query",
            model,
            system_prompt="Transform the query.",
            temperature=0.0,
            top_p=1.0,
            max_tokens=128,
        )

    result = anyio.run(collect)

    assert result == "transformed query"
    assert "private provider query" not in caplog.text
    assert "query_len=" in caplog.text
    assert "model=" in caplog.text
