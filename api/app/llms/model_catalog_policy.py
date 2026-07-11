from dataclasses import dataclass
from typing import Final

from app.llms.model_catalog_types import CatalogProvider, CatalogTier, ExecutionPolicy
from app.llms.models import (
    ANTHROPIC_NO_SAMPLING_MODELS,
    CHAT_MODEL_ORDER,
    REASONING_CAPABLE_MODELS,
    Model,
)


@dataclass(frozen=True, slots=True)
class CatalogPolicy:
    model: Model
    provider: CatalogProvider
    tier: CatalogTier
    display_name: str
    execution: ExecutionPolicy


def _policy(
    model: Model,
    provider: CatalogProvider,
    tier: CatalogTier,
    display_name: str,
) -> CatalogPolicy:
    return CatalogPolicy(
        model=model,
        provider=provider,
        tier=tier,
        display_name=display_name,
        execution=ExecutionPolicy(
            reasoning=model in REASONING_CAPABLE_MODELS,
            sampling=model not in ANTHROPIC_NO_SAMPLING_MODELS,
            tool_call=provider != "ollama",
            structured_output=provider != "ollama",
        ),
    )


MODEL_CATALOG: Final[tuple[CatalogPolicy, ...]] = (
    _policy(Model.GPT_5_6_SOL, "openai", "premium", "GPT-5.6 Sol"),
    _policy(Model.GPT_5_6_TERRA, "openai", "default", "GPT-5.6 Terra"),
    _policy(Model.GPT_5_6_LUNA, "openai", "cheap", "GPT-5.6 Luna"),
    _policy(Model.GPT_5_5, "openai", "premium", "GPT-5.5"),
    _policy(Model.GPT_5_4, "openai", "premium", "GPT-5.4"),
    _policy(Model.GPT_5_4_MINI, "openai", "default", "GPT-5.4 Mini"),
    _policy(Model.GPT_5_4_NANO, "openai", "cheap", "GPT-5.4 Nano"),
    _policy(
        Model.GEMINI_3_1_PRO_PREVIEW,
        "google",
        "premium",
        "Gemini 3.1 Pro Preview",
    ),
    _policy(Model.GEMINI_3_5_FLASH, "google", "default", "Gemini 3.5 Flash"),
    _policy(
        Model.GEMINI_3_1_FLASH_LITE,
        "google",
        "cheap",
        "Gemini 3.1 Flash Lite",
    ),
    _policy(Model.CLAUDE_OPUS_4_8, "anthropic", "premium", "Claude Opus 4.8"),
    _policy(Model.CLAUDE_SONNET_5, "anthropic", "default", "Claude Sonnet 5"),
    _policy(Model.CLAUDE_HAIKU_4_5, "anthropic", "cheap", "Claude Haiku 4.5"),
    _policy(
        Model.QWEN3_30B_THINKING,
        "ollama",
        "premium",
        "Qwen3 30B Thinking",
    ),
    _policy(
        Model.QWEN3_30B_INSTRUCT,
        "ollama",
        "default",
        "Qwen3 30B Instruct",
    ),
    _policy(Model.QWEN3_14B, "ollama", "cheap", "Qwen3 14B"),
)

if tuple(policy.model for policy in MODEL_CATALOG) != CHAT_MODEL_ORDER:
    msg = "Catalog policy order must match the executable chat model order"
    raise RuntimeError(msg)
