import pytest

from app.llms.models import Model
from app.llms.pricing import HOSTED_PRICING, cost_usd


@pytest.mark.parametrize(
    ("model", "expected_price"),
    [
        (Model.GPT_6_ASTRA, (10.00, 50.00)),
        (Model.GEMINI_3_8_FLASH, (0.75, 3.75)),
        (Model.CLAUDE_FABLE_5_1, (10.00, 50.00)),
        (Model.OPENROUTER_DEEPSEEK_V4_1_FLASH, (0.15, 0.60)),
        (Model.OPENROUTER_QWEN3_8_MAX_0902, (2.00, 6.00)),
    ],
)
def test_new_curated_models_have_static_token_price_estimates(
    model: Model,
    expected_price: tuple[float, float],
) -> None:
    assert HOSTED_PRICING[model] == expected_price
    assert cost_usd(model, 1_000_000, 1_000_000) == (
        expected_price[0],
        expected_price[1],
        sum(expected_price),
    )


@pytest.mark.parametrize(
    "model",
    [
        Model.GPT_6_1_SOL,
        Model.GPT_6_LUNA,
        Model.CLAUDE_HAIKU_5_5,
        Model.OPENROUTER_GROK_4_7,
    ],
)
def test_replacement_models_without_verified_prices_remain_unpriced(
    model: Model,
) -> None:
    assert model not in HOSTED_PRICING
    assert cost_usd(model, 1_000_000, 1_000_000) is None
