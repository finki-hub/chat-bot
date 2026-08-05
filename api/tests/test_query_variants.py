import anyio
import pytest
from anyio.lowlevel import checkpoint

from app.llms import query_variants
from app.llms.models import Model
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
        (RuntimeError("rewrite failed"), "Хипотетички пасус", ("hyde", "raw"), "query"),
        ("rewritten query", RuntimeError("hyde failed"), ("rewrite", "raw"), "rewritten query"),
        (RuntimeError("rewrite failed"), RuntimeError("hyde failed"), ("raw",), "query"),
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
