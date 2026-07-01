from enum import Enum


class Model(Enum):
    """
    Enum representing the available models for inference and embeddings.
    """

    LLAMA_3_3_70B = "llama3.3:70b"
    MISTRAL = "mistral:latest"
    DEEPSEEK_R1_70B = "deepseek-r1:70b"
    QWEN2_5_72B = "qwen2.5:72b"
    BGE_M3 = "bge-m3:latest"

    GPT_5_2 = "gpt-5.2"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_NANO = "gpt-5-nano"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_1 = "gpt-4.1"
    GPT_4_1_MINI = "gpt-4.1-mini"
    GPT_4_1_NANO = "gpt-4.1-nano"
    GPT_5_4 = "gpt-5.4"
    GPT_5_4_MINI = "gpt-5.4-mini"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"

    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_3_FLASH_PREVIEW = "gemini-3-flash-preview"
    GEMINI_EMBEDDING_001 = "gemini-embedding-001"

    CLAUDE_OPUS_4_8 = "claude-opus-4-8"
    CLAUDE_OPUS_4_7 = "claude-opus-4-7"
    CLAUDE_SONNET_5 = "claude-sonnet-5"
    CLAUDE_SONNET_4_6 = "claude-sonnet-4-6"
    CLAUDE_HAIKU_4_5 = "claude-haiku-4-5"

    DOMESTIC_YAK_8B_INSTRUCT_GGUF = "hf.co/LVSTCK/domestic-yak-8B-instruct-GGUF:Q8_0"
    VEZILKALLM_GGUF = "hf.co/mradermacher/VezilkaLLM-GGUF:Q8_0"

    MULTILINGUAL_E5_LARGE = "intfloat/multilingual-e5-large"
    BGE_M3_LOCAL = "BAAI/bge-m3"
    QWEN2_1_5_B_INSTRUCT = "Qwen/Qwen2-1.5B-Instruct"
    QWEN2_5_7B_INSTRUCT = "Qwen/Qwen2.5-7B-Instruct"


MODEL_EMBEDDINGS_COLUMNS: dict[Model, str] = {
    Model.LLAMA_3_3_70B: "embedding_llama3_3_70b",
    Model.BGE_M3: "embedding_bge_m3",
    Model.BGE_M3_LOCAL: "embedding_bge_m3",
    Model.TEXT_EMBEDDING_3_LARGE: "embedding_text_embedding_3_large",
    Model.GEMINI_EMBEDDING_001: "embedding_gemini_embedding_001",
    Model.MULTILINGUAL_E5_LARGE: "embedding_multilingual_e5_large",
}

# Models filled by an `all_models` request: exactly one per *indexed, retrievable*
# embedding column. Excludes LLAMA_3_3_70B (8192 dims -> unindexed, so it can never be
# used for ANN retrieval) and resolves the BGE_M3 / BGE_M3_LOCAL pair (both target
# embedding_bge_m3) to the gpu-api BGE_M3_LOCAL provider, so a column is never filled
# twice. Without this, all_models would double-fill embedding_bge_m3 and waste compute
# on the un-retrievable llama column.
ALL_MODELS_EMBEDDINGS: tuple[Model, ...] = (
    Model.BGE_M3_LOCAL,
    Model.TEXT_EMBEDDING_3_LARGE,
    Model.GEMINI_EMBEDDING_001,
    Model.MULTILINGUAL_E5_LARGE,
)

GPU_API_MODELS: dict[Model, str] = {
    Model.BGE_M3_LOCAL: "BAAI/bge-m3",
    Model.MULTILINGUAL_E5_LARGE: "intfloat/multilingual-e5-large",
    Model.QWEN2_1_5_B_INSTRUCT: "Qwen/Qwen2-1.5B-Instruct",
    Model.QWEN2_5_7B_INSTRUCT: "Qwen/Qwen2.5-7B-Instruct",
}

CHAT_MODELS: frozenset[Model] = frozenset(
    {
        Model.LLAMA_3_3_70B,
        Model.MISTRAL,
        Model.DEEPSEEK_R1_70B,
        Model.QWEN2_5_72B,
        Model.DOMESTIC_YAK_8B_INSTRUCT_GGUF,
        Model.VEZILKALLM_GGUF,
        Model.GPT_4O_MINI,
        Model.GPT_4_1,
        Model.GPT_4_1_MINI,
        Model.GPT_4_1_NANO,
        Model.GPT_5_4,
        Model.GPT_5_4_MINI,
        Model.GPT_5_2,
        Model.GPT_5_MINI,
        Model.GPT_5_NANO,
        Model.GEMINI_2_5_FLASH,
        Model.GEMINI_2_5_PRO,
        Model.GEMINI_3_FLASH_PREVIEW,
        Model.CLAUDE_OPUS_4_8,
        Model.CLAUDE_OPUS_4_7,
        Model.CLAUDE_SONNET_5,
        Model.CLAUDE_SONNET_4_6,
        Model.CLAUDE_HAIKU_4_5,
        Model.QWEN2_1_5_B_INSTRUCT,
        Model.QWEN2_5_7B_INSTRUCT,
    },
)

OPENAI_QUERY_TRANSFORM_MODELS: frozenset[Model] = frozenset(
    {
        Model.GPT_4O_MINI,
        Model.GPT_4_1,
        Model.GPT_4_1_MINI,
        Model.GPT_4_1_NANO,
        Model.GPT_5_4,
        Model.GPT_5_4_MINI,
        Model.GPT_5_2,
        Model.GPT_5_MINI,
        Model.GPT_5_NANO,
    },
)
GOOGLE_QUERY_TRANSFORM_MODELS: frozenset[Model] = frozenset(
    {
        Model.GEMINI_2_5_FLASH,
        Model.GEMINI_2_5_PRO,
        Model.GEMINI_3_FLASH_PREVIEW,
    },
)
ANTHROPIC_QUERY_TRANSFORM_MODELS: frozenset[Model] = frozenset(
    {
        Model.CLAUDE_OPUS_4_8,
        Model.CLAUDE_OPUS_4_7,
        Model.CLAUDE_SONNET_5,
        Model.CLAUDE_SONNET_4_6,
        Model.CLAUDE_HAIKU_4_5,
    },
)
OLLAMA_QUERY_TRANSFORM_MODELS: frozenset[Model] = frozenset(
    {
        Model.LLAMA_3_3_70B,
        Model.MISTRAL,
        Model.DEEPSEEK_R1_70B,
        Model.QWEN2_5_72B,
        Model.DOMESTIC_YAK_8B_INSTRUCT_GGUF,
        Model.VEZILKALLM_GGUF,
    },
)
QUERY_TRANSFORM_MODELS: frozenset[Model] = frozenset(
    {
        *OPENAI_QUERY_TRANSFORM_MODELS,
        *GOOGLE_QUERY_TRANSFORM_MODELS,
        *ANTHROPIC_QUERY_TRANSFORM_MODELS,
        *OLLAMA_QUERY_TRANSFORM_MODELS,
    },
)

REASONING_CAPABLE_MODELS: frozenset[Model] = frozenset(
    {
        Model.DEEPSEEK_R1_70B,
        Model.GPT_5_4,
        Model.GPT_5_4_MINI,
        Model.GPT_5_2,
        Model.GPT_5_MINI,
        Model.GPT_5_NANO,
        Model.GEMINI_2_5_FLASH,
        Model.GEMINI_2_5_PRO,
        Model.GEMINI_3_FLASH_PREVIEW,
        Model.CLAUDE_OPUS_4_8,
        Model.CLAUDE_OPUS_4_7,
        Model.CLAUDE_SONNET_5,
        Model.CLAUDE_SONNET_4_6,
        Model.CLAUDE_HAIKU_4_5,
    },
)

# OpenAI reasoning models that default to "medium" effort and, in a non-reasoning chat,
# can spend the whole max_output_tokens budget on hidden reasoning and return an empty
# answer (observed for gpt-5-mini / gpt-5-nano). They accept reasoning effort "minimal",
# which keeps them answering. The newer GPT-5.2 / 5.4 models reject "minimal" and do not
# over-reason at the default, so they are deliberately excluded.
OPENAI_MINIMAL_EFFORT_MODELS: frozenset[Model] = frozenset(
    {
        Model.GPT_5_MINI,
        Model.GPT_5_NANO,
    },
)

# Anthropic models that reject sampling parameters (temperature / top_p / top_k).
# Claude Opus 4.7 and 4.8 return HTTP 400 if any of these are sent, so no sampling
# parameters are forwarded for them. Every Claude 4+ model additionally rejects
# requests that set both temperature and top_p, which is why the Anthropic client
# never forwards top_p for any Claude model.
ANTHROPIC_NO_SAMPLING_MODELS: frozenset[Model] = frozenset(
    {
        Model.CLAUDE_OPUS_4_8,
        Model.CLAUDE_OPUS_4_7,
    },
)

HALFVEC_EMBEDDING_MODELS: frozenset[Model] = frozenset(
    {
        Model.TEXT_EMBEDDING_3_LARGE,
        Model.GEMINI_EMBEDDING_001,
    },
)

# Cosine-distance ceiling for the ANN pre-filter, per embedding model — a coarse RECALL
# gate, not the precision gate (the cross-encoder reranker / RERANKER_MIN_SCORE is that),
# so it errs toward letting candidates through. Kept generous on purpose.
#
# Only the bge-m3 values are eval-calibrated: the harness (api/tests/eval) found relevant
# bge-m3 sources at ~0.40-0.55 that the old 0.40 ceiling dropped before the reranker saw
# them (the "praksa" miss). The other models share the same 0.5 as an untuned carry-over —
# their cosine-distance distributions differ, so recalibrate each with
# `run_eval.py --embedding-model <name>` before trusting the ceiling for that model.
MODEL_DISTANCE_THRESHOLDS: dict[Model, float] = {
    Model.BGE_M3: 0.5,
    Model.BGE_M3_LOCAL: 0.5,
    Model.MULTILINGUAL_E5_LARGE: 0.5,
    Model.TEXT_EMBEDDING_3_LARGE: 0.5,
    Model.GEMINI_EMBEDDING_001: 0.5,
    Model.LLAMA_3_3_70B: 0.5,
}

MODEL_EMBEDDING_DIMENSIONS: dict[Model, int] = {
    Model.LLAMA_3_3_70B: 8192,
    Model.BGE_M3: 1024,
    Model.BGE_M3_LOCAL: 1024,
    Model.TEXT_EMBEDDING_3_LARGE: 3072,
    Model.GEMINI_EMBEDDING_001: 3072,
    Model.MULTILINGUAL_E5_LARGE: 1024,
}


def _validate_model_maps() -> None:
    """Fail fast at import on a silently-wrong embedding configuration.

    A retrievable model with >2000 dims that is missing from HALFVEC_EMBEDDING_MODELS
    builds no usable index (pgvector caps a vector HNSW index at 2000 dims) and silently
    degrades to a sequential scan. Any per-model map keyed on a model with no embedding
    column is dead configuration.
    """
    for model in ALL_MODELS_EMBEDDINGS:
        dims = MODEL_EMBEDDING_DIMENSIONS.get(model)
        if dims is not None and dims > 2000 and model not in HALFVEC_EMBEDDING_MODELS:
            msg = (
                f"{model.value} has {dims} dims but is not in HALFVEC_EMBEDDING_MODELS; "
                "its index would exceed pgvector's 2000-dim vector limit and be unused"
            )
            raise RuntimeError(msg)

    columns = set(MODEL_EMBEDDINGS_COLUMNS)
    for name, mapping in (
        ("HALFVEC_EMBEDDING_MODELS", HALFVEC_EMBEDDING_MODELS),
        ("MODEL_DISTANCE_THRESHOLDS", MODEL_DISTANCE_THRESHOLDS),
        ("MODEL_EMBEDDING_DIMENSIONS", MODEL_EMBEDDING_DIMENSIONS),
    ):
        unknown = set(mapping) - columns
        if unknown:
            names = ", ".join(sorted(m.value for m in unknown))
            msg = f"{name} references models with no embedding column: {names}"
            raise RuntimeError(msg)


_validate_model_maps()
