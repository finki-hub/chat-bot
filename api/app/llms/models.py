from enum import StrEnum
from typing import Final


class Model(StrEnum):
    LLAMA_3_3_70B = "llama3.3:70b"
    DEEPSEEK_R1_70B = "deepseek-r1:70b"
    BGE_M3 = "bge-m3:latest"

    GPT_5_4 = "gpt-5.4"
    GPT_5_4_MINI = "gpt-5.4-mini"
    GPT_5_4_NANO = "gpt-5.4-nano"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"

    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_EMBEDDING_001 = "gemini-embedding-001"

    CLAUDE_OPUS_4_8 = "claude-opus-4-8"
    CLAUDE_SONNET_5 = "claude-sonnet-5"
    CLAUDE_HAIKU_4_5 = "claude-haiku-4-5"

    DOMESTIC_YAK_8B_INSTRUCT_GGUF = "hf.co/LVSTCK/domestic-yak-8B-instruct-GGUF:Q8_0"
    VEZILKALLM_GGUF = "hf.co/mradermacher/VezilkaLLM-GGUF:Q8_0"

    MULTILINGUAL_E5_LARGE = "intfloat/multilingual-e5-large"
    BGE_M3_LOCAL = "BAAI/bge-m3"


MODEL_EMBEDDINGS_COLUMNS: Final[dict[Model, str]] = {
    Model.LLAMA_3_3_70B: "embedding_llama3_3_70b",
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
    Model.GPT_5_4,
    Model.GPT_5_4_MINI,
    Model.GPT_5_4_NANO,
    Model.GEMINI_2_5_PRO,
    Model.GEMINI_2_5_FLASH,
    Model.CLAUDE_OPUS_4_8,
    Model.CLAUDE_SONNET_5,
    Model.CLAUDE_HAIKU_4_5,
    Model.LLAMA_3_3_70B,
    Model.DEEPSEEK_R1_70B,
    Model.DOMESTIC_YAK_8B_INSTRUCT_GGUF,
    Model.VEZILKALLM_GGUF,
)
CHAT_MODELS: Final[frozenset[Model]] = frozenset(CHAT_MODEL_ORDER)

OPENAI_QUERY_TRANSFORM_MODELS: Final[frozenset[Model]] = frozenset(
    {Model.GPT_5_4, Model.GPT_5_4_MINI, Model.GPT_5_4_NANO},
)
GOOGLE_QUERY_TRANSFORM_MODELS: Final[frozenset[Model]] = frozenset(
    {Model.GEMINI_2_5_FLASH, Model.GEMINI_2_5_PRO},
)
ANTHROPIC_QUERY_TRANSFORM_MODELS: Final[frozenset[Model]] = frozenset(
    {Model.CLAUDE_OPUS_4_8, Model.CLAUDE_SONNET_5, Model.CLAUDE_HAIKU_4_5},
)
OLLAMA_QUERY_TRANSFORM_MODELS: Final[frozenset[Model]] = frozenset(
    {
        Model.LLAMA_3_3_70B,
        Model.DEEPSEEK_R1_70B,
        Model.DOMESTIC_YAK_8B_INSTRUCT_GGUF,
        Model.VEZILKALLM_GGUF,
    },
)
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
        Model.GPT_5_4,
        Model.GPT_5_4_MINI,
        Model.GPT_5_4_NANO,
        Model.GEMINI_2_5_FLASH,
        Model.GEMINI_2_5_PRO,
        Model.CLAUDE_OPUS_4_8,
        Model.CLAUDE_SONNET_5,
        Model.CLAUDE_HAIKU_4_5,
        Model.DEEPSEEK_R1_70B,
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
    Model.LLAMA_3_3_70B: 8192,
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
