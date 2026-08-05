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
