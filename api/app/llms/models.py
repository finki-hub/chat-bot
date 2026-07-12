from enum import StrEnum
from typing import Final


class Model(StrEnum):
    BGE_M3 = "bge-m3:latest"

    GPT_5_6_SOL = "gpt-5.6-sol"
    GPT_5_6_TERRA = "gpt-5.6-terra"
    GPT_5_6_LUNA = "gpt-5.6-luna"
    GPT_5_5 = "gpt-5.5"
    GPT_5_4 = "gpt-5.4"
    GPT_5_4_MINI = "gpt-5.4-mini"
    GPT_5_4_NANO = "gpt-5.4-nano"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"

    GEMINI_3_1_PRO_PREVIEW = "gemini-3.1-pro-preview"
    GEMINI_3_5_FLASH = "gemini-3.5-flash"
    GEMINI_3_1_FLASH_LITE = "gemini-3.1-flash-lite"
    GEMINI_EMBEDDING_001 = "gemini-embedding-001"

    CLAUDE_OPUS_4_8 = "claude-opus-4-8"
    CLAUDE_SONNET_5 = "claude-sonnet-5"
    CLAUDE_HAIKU_4_5 = "claude-haiku-4-5"

    QWEN3_30B_THINKING = "qwen3:30b-a3b-thinking-2507-q4_K_M"
    QWEN3_30B_INSTRUCT = "qwen3:30b-a3b-instruct-2507-q4_K_M"
    QWEN3_14B = "qwen3:14b-q4_K_M"

    MULTILINGUAL_E5_LARGE = "intfloat/multilingual-e5-large"
    BGE_M3_LOCAL = "BAAI/bge-m3"


MODEL_EMBEDDINGS_COLUMNS: Final[dict[Model, str]] = {
    Model.BGE_M3: "embedding_bge_m3",
    Model.BGE_M3_LOCAL: "embedding_bge_m3",
    Model.TEXT_EMBEDDING_3_LARGE: "embedding_text_embedding_3_large",
    Model.GEMINI_EMBEDDING_001: "embedding_gemini_embedding_001",
    Model.MULTILINGUAL_E5_LARGE: "embedding_multilingual_e5_large",
}

ACTIVE_EMBEDDING_MODELS: Final[frozenset[Model]] = frozenset(
    {
        Model.BGE_M3_LOCAL,
        Model.GEMINI_EMBEDDING_001,
        Model.TEXT_EMBEDDING_3_LARGE,
    },
)
ALL_MODELS_EMBEDDINGS: Final[tuple[Model, ...]] = (Model.BGE_M3_LOCAL,)
GPU_API_MODELS: Final[dict[Model, str]] = {Model.BGE_M3_LOCAL: "BAAI/bge-m3"}

CHAT_MODEL_ORDER: Final[tuple[Model, ...]] = (
    Model.GPT_5_6_SOL,
    Model.GPT_5_6_TERRA,
    Model.GPT_5_6_LUNA,
    Model.GPT_5_5,
    Model.GPT_5_4,
    Model.GPT_5_4_MINI,
    Model.GPT_5_4_NANO,
    Model.GEMINI_3_1_PRO_PREVIEW,
    Model.GEMINI_3_5_FLASH,
    Model.GEMINI_3_1_FLASH_LITE,
    Model.CLAUDE_OPUS_4_8,
    Model.CLAUDE_SONNET_5,
    Model.CLAUDE_HAIKU_4_5,
)
CHAT_MODELS: Final[frozenset[Model]] = frozenset(CHAT_MODEL_ORDER)
type ChatModel = Model | str

OPENAI_QUERY_TRANSFORM_MODELS: Final[frozenset[Model]] = frozenset(
    {
        Model.GPT_5_6_SOL,
        Model.GPT_5_6_TERRA,
        Model.GPT_5_6_LUNA,
        Model.GPT_5_5,
        Model.GPT_5_4,
        Model.GPT_5_4_MINI,
        Model.GPT_5_4_NANO,
    },
)
GOOGLE_QUERY_TRANSFORM_MODELS: Final[frozenset[Model]] = frozenset(
    {
        Model.GEMINI_3_1_PRO_PREVIEW,
        Model.GEMINI_3_5_FLASH,
        Model.GEMINI_3_1_FLASH_LITE,
    },
)
ANTHROPIC_QUERY_TRANSFORM_MODELS: Final[frozenset[Model]] = frozenset(
    {Model.CLAUDE_OPUS_4_8, Model.CLAUDE_SONNET_5, Model.CLAUDE_HAIKU_4_5},
)
OLLAMA_QUERY_TRANSFORM_MODELS: Final[frozenset[Model]] = frozenset()
QUERY_TRANSFORM_MODELS: Final[frozenset[Model]] = frozenset(
    {
        *OPENAI_QUERY_TRANSFORM_MODELS,
        *GOOGLE_QUERY_TRANSFORM_MODELS,
        *ANTHROPIC_QUERY_TRANSFORM_MODELS,
        *OLLAMA_QUERY_TRANSFORM_MODELS,
    },
)

REASONING_CAPABLE_MODELS: Final[frozenset[Model]] = frozenset(
    {
        Model.GPT_5_6_SOL,
        Model.GPT_5_6_TERRA,
        Model.GPT_5_6_LUNA,
        Model.GPT_5_5,
        Model.GPT_5_4,
        Model.GPT_5_4_MINI,
        Model.GPT_5_4_NANO,
        Model.GEMINI_3_1_PRO_PREVIEW,
        Model.GEMINI_3_5_FLASH,
        Model.GEMINI_3_1_FLASH_LITE,
        Model.CLAUDE_OPUS_4_8,
        Model.CLAUDE_SONNET_5,
        Model.CLAUDE_HAIKU_4_5,
    },
)
OPENAI_MINIMAL_EFFORT_MODELS: Final[frozenset[Model]] = frozenset()
ANTHROPIC_NO_SAMPLING_MODELS: Final[frozenset[Model]] = frozenset(
    {Model.CLAUDE_OPUS_4_8, Model.CLAUDE_SONNET_5},
)

HALFVEC_EMBEDDING_MODELS: Final[frozenset[Model]] = frozenset(
    {Model.TEXT_EMBEDDING_3_LARGE, Model.GEMINI_EMBEDDING_001},
)
MODEL_DISTANCE_THRESHOLDS: Final[dict[Model, float]] = dict.fromkeys(
    MODEL_EMBEDDINGS_COLUMNS,
    0.5,
)
MODEL_EMBEDDING_DIMENSIONS: Final[dict[Model, int]] = {
    Model.BGE_M3: 1024,
    Model.BGE_M3_LOCAL: 1024,
    Model.TEXT_EMBEDDING_3_LARGE: 3072,
    Model.GEMINI_EMBEDDING_001: 3072,
    Model.MULTILINGUAL_E5_LARGE: 1024,
}


def _validate_model_maps() -> None:
    columns = set(MODEL_EMBEDDINGS_COLUMNS)
    for mapping in (
        HALFVEC_EMBEDDING_MODELS,
        frozenset(MODEL_DISTANCE_THRESHOLDS),
        frozenset(MODEL_EMBEDDING_DIMENSIONS),
    ):
        unknown = set(mapping) - columns
        if unknown:
            names = ", ".join(sorted(model.value for model in unknown))
            msg = f"Embedding configuration references models without columns: {names}"
            raise RuntimeError(msg)


_validate_model_maps()


def model_id(model: ChatModel) -> str:
    return model.value if isinstance(model, Model) else model


def is_ollama_model_tag(value: str) -> bool:
    stripped = value.strip()
    return stripped == value and ":" in value and bool(value)


def parse_chat_model(value: Model | str, active_models: frozenset[Model]) -> ChatModel:
    if isinstance(value, Model):
        if value in active_models:
            return value
        raise ValueError("model must be an active chat model")
    try:
        model = Model(value)
    except ValueError:
        if is_ollama_model_tag(value):
            return value
        raise ValueError("model must be an active chat model") from None
    if model in active_models:
        return model
    if is_ollama_model_tag(value):
        return value
    raise ValueError("model must be an active chat model")
