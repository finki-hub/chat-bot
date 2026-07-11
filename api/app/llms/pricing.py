"""Single source of truth for hosted-model token prices (USD per 1M tokens).

Update prices here when a provider changes them. Self-hosted local GPU models have no
marginal token cost and are priced at 0; provider models without a reliable published price
are omitted, so callers treat their cost as unknown rather than guessing.
"""

from app.llms.models import Model

# (input_usd_per_1m, output_usd_per_1m) for hosted models with a reliable published price.
HOSTED_PRICING: dict[Model, tuple[float, float]] = {
    Model.GPT_5_6_SOL: (5.00, 30.00),
    Model.GPT_5_6_TERRA: (2.50, 15.00),
    Model.GPT_5_6_LUNA: (1.00, 6.00),
    Model.GPT_5_5: (5.00, 30.00),
    Model.GPT_5_4: (2.50, 15.00),
    Model.GPT_5_4_MINI: (0.75, 4.50),
    Model.GPT_5_4_NANO: (0.20, 1.25),
    Model.GEMINI_3_1_PRO_PREVIEW: (2.00, 12.00),
    Model.GEMINI_3_5_FLASH: (1.50, 9.00),
    Model.GEMINI_3_1_FLASH_LITE: (0.25, 1.50),
    Model.CLAUDE_OPUS_4_8: (5.00, 25.00),
    Model.CLAUDE_SONNET_5: (3.00, 15.00),
    Model.CLAUDE_HAIKU_4_5: (1.00, 5.00),
}

# Models we run ourselves on the local GPU: zero marginal token cost.
SELF_HOSTED_MODELS: frozenset[Model] = frozenset(
    {
        Model.BGE_M3_LOCAL,
    },
)


def is_self_hosted(model: Model) -> bool:
    return model in SELF_HOSTED_MODELS


def cost_usd(
    model: Model,
    input_tokens: int,
    output_tokens: int,
) -> tuple[float, float, float] | None:
    """(input, output, total) USD for a generation, or None when the price is unknown.

    Self-hosted models cost 0; an unpriced hosted model returns None so the caller can flag
    the cost as unknown instead of recording a fabricated 0.
    """
    if model in SELF_HOSTED_MODELS:
        return (0.0, 0.0, 0.0)
    price = HOSTED_PRICING.get(model)
    if price is None:
        return None
    input_cost = input_tokens / 1_000_000 * price[0]
    output_cost = output_tokens / 1_000_000 * price[1]
    return (input_cost, output_cost, input_cost + output_cost)
