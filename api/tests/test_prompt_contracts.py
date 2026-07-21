import anyio
import pytest
from anyio import lowlevel

from app.llms import query_variants
from app.llms.models import Model
from app.llms.provider_credentials import LlmProviderCredentials
from app.llms.query_modes import QueryTransformMode


def test_hyde_uses_conservative_sampling(monkeypatch):
    seen: dict[str, float | str | Model] = {}

    async def fake_transform_query(
        query: str,
        model: Model,
        *,
        system_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        credentials: LlmProviderCredentials | None = None,
    ) -> str:
        await lowlevel.checkpoint()
        seen.update(
            {
                "max_tokens": max_tokens,
                "model": model,
                "query": query,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "top_p": top_p,
            },
        )
        return "Пасус"

    monkeypatch.setattr(query_variants, "transform_query", fake_transform_query)

    result = anyio.run(
        query_variants.build_query_variants,
        "Како се пријавува испит?",
        Model.GPT_5_4_MINI,
        QueryTransformMode.HYDE,
    )

    assert result.variants[0].text == "Пасус"
    assert seen["temperature"] == pytest.approx(0.2)
    assert seen["top_p"] == pytest.approx(1.0)
