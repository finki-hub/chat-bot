"""Single source of truth for hosted-model token prices (USD per 1M tokens).

Update prices here when a provider changes them. Self-hosted local GPU models have no
marginal token cost and are priced at 0; provider models without a reliable published price
are omitted, so callers treat their cost as unknown rather than guessing.
"""

from app.llms.models import Model

# (input_usd_per_1m, output_usd_per_1m) for hosted models with a reliable published price.
HOSTED_PRICING: dict[Model, tuple[float, float]] = {
    Model.GPT_4O_MINI: (0.15, 0.60),
    Model.GPT_4_1: (2.00, 8.00),
    Model.GPT_4_1_MINI: (0.40, 1.60),
    Model.GPT_4_1_NANO: (0.10, 0.40),
    Model.GEMINI_2_5_FLASH: (0.30, 2.50),
    Model.GEMINI_2_5_PRO: (1.25, 10.00),
    Model.CLAUDE_HAIKU_4_5: (1.00, 5.00),
    Model.CLAUDE_SONNET_4_6: (3.00, 15.00),
}

# Models we run ourselves on the local GPU: zero marginal token cost.
SELF_HOSTED_MODELS: frozenset[Model] = frozenset(
    {
        Model.MULTILINGUAL_E5_LARGE,
        Model.BGE_M3_LOCAL,
        Model.QWEN2_1_5_B_INSTRUCT,
        Model.QWEN2_5_7B_INSTRUCT,
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
