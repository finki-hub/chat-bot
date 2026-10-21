import anyio
import pytest
from anyio.lowlevel import checkpoint

from app.llms import query_variants
from app.llms.models import Model
from app.llms.prompts import HYDE_SYSTEM_PROMPT
from app.llms.query_modes import QueryTransformMode


def test_rewrite_hyde_omits_variants_unchanged_from_search_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_query = "Како се пријавува испит?"

    async def unchanged_transform(*args, **kwargs):
        await checkpoint()
        return search_query

    monkeypatch.setattr(query_variants, "transform_query", unchanged_transform)

    result = anyio.run(
        query_variants.build_query_variants,
        search_query,
        Model.GPT_5_4_MINI,
        QueryTransformMode.REWRITE_HYDE,
    )

    assert result.variants == (
        query_variants.QueryVariant(
            kind="raw",
            text=search_query,
            is_document=False,
        ),
    )
    assert result.rerank_query == search_query


@pytest.mark.parametrize(
    ("rewrite_result", "hyde_result", "expected_kinds", "expected_rerank_query"),
    [
        (
            "rewritten query",
            "Хипотетички пасус",
            ("hyde", "rewrite", "raw"),
            "query",
        ),
        (RuntimeError("rewrite failed"), "Хипотетички пасус", ("hyde", "raw"), "query"),
        (
            "rewritten query",
            RuntimeError("hyde failed"),
            ("rewrite", "raw"),
            "query",
        ),
        (
            RuntimeError("rewrite failed"),
            RuntimeError("hyde failed"),
            ("raw",),
            "query",
        ),
    ],
)
def test_rewrite_hyde_omits_each_failed_transform_independently(
    monkeypatch: pytest.MonkeyPatch,
    rewrite_result: str | RuntimeError,
    hyde_result: str | RuntimeError,
    expected_kinds: tuple[str, ...],
    expected_rerank_query: str,
) -> None:
    async def transform(*args, **kwargs):
        await checkpoint()
        result = hyde_result if "system_prompt" in kwargs else rewrite_result
        if isinstance(result, RuntimeError):
            raise result
        return result

    monkeypatch.setattr(query_variants, "transform_query", transform)

    result = anyio.run(
        query_variants.build_query_variants,
        "query",
        Model.GPT_5_4_MINI,
        QueryTransformMode.REWRITE_HYDE,
    )

    assert tuple(variant.kind for variant in result.variants) == expected_kinds
    assert result.rerank_query == expected_rerank_query


def test_rewrite_hyde_propagates_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def cancel(*args, **kwargs):
        await checkpoint()
        raise anyio.get_cancelled_exc_class()

    monkeypatch.setattr(query_variants, "transform_query", cancel)

    async def collect() -> None:
        with pytest.raises(anyio.get_cancelled_exc_class()):
            await query_variants.build_query_variants(
                "query",
                Model.GPT_5_4_MINI,
                QueryTransformMode.REWRITE_HYDE,
            )

    anyio.run(collect)


@pytest.mark.parametrize(
    ("mode", "expected_kinds", "expected_rerank_query"),
    [
        (QueryTransformMode.RAW, ("transliteration", "raw"), "Каде е слузба"),
        (
            QueryTransformMode.REWRITE,
            ("rewrite", "transliteration", "raw"),
            "Каде е слузба",
        ),
        (
            QueryTransformMode.HYDE,
            ("hyde", "transliteration", "raw"),
            "Каде е слузба",
        ),
    ],
)
def test_romanized_macedonian_query_adds_cyrillic_variant_in_every_mode(
    monkeypatch: pytest.MonkeyPatch,
    mode: QueryTransformMode,
    expected_kinds: tuple[str, ...],
    expected_rerank_query: str,
) -> None:
    async def transform(*args, **kwargs):
        await checkpoint()
        if "system_prompt" in kwargs:
            return "Студентската служба се наоѓа во приземје."
        return "Каде се наоѓа Студентската служба?"

    monkeypatch.setattr(query_variants, "transform_query", transform)

    result = anyio.run(
        query_variants.build_query_variants,
        "Kade e sluzba",
        Model.GPT_5_4_MINI,
        mode,
    )

    assert tuple(variant.kind for variant in result.variants) == expected_kinds
    assert result.variants[-2] == query_variants.QueryVariant(
        kind="transliteration",
        text="Каде е слузба",
        is_document=False,
    )
    assert result.transliteration_variant_count == 1
    assert result.mode == mode
    assert result.rerank_query == expected_rerank_query


def test_english_query_adds_no_transliteration_variant() -> None:
    result = anyio.run(
        query_variants.build_query_variants,
        "Web programming",
        Model.GPT_5_4_MINI,
        QueryTransformMode.RAW,
    )

    assert tuple(variant.kind for variant in result.variants) == ("raw",)
    assert result.transliteration_variant_count == 0
    assert result.rerank_query == "Web programming"


def test_romanized_query_keeps_unchanged_rewrite_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_query = "kade e studentska sluzba"

    async def transform(*args, **kwargs):
        await checkpoint()
        if kwargs.get("system_prompt") == HYDE_SYSTEM_PROMPT:
            return "Студентската служба се наоѓа во зградата на ФИНКИ."
        return search_query

    monkeypatch.setattr(query_variants, "transform_query", transform)

    result = anyio.run(
        query_variants.build_query_variants,
        search_query,
        Model.GPT_5_4_MINI,
        QueryTransformMode.REWRITE_HYDE,
    )

    assert [variant.kind for variant in result.variants] == [
        "hyde",
        "transliteration",
        "raw",
    ]
    assert result.unchanged_kinds == frozenset({"rewrite"})
    assert (
        result.fallback_reason(QueryTransformMode.REWRITE_HYDE) == "unchanged_variants"
    )
