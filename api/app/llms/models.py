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
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"

    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_3_FLASH_PREVIEW = "gemini-3-flash-preview"
    GEMINI_EMBEDDING_001 = "gemini-embedding-001"

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

GPU_API_MODELS: dict[Model, str] = {
    Model.BGE_M3_LOCAL: "BAAI/bge-m3",
    Model.MULTILINGUAL_E5_LARGE: "intfloat/multilingual-e5-large",
    Model.QWEN2_1_5_B_INSTRUCT: "Qwen/Qwen2-1.5B-Instruct",
    Model.QWEN2_5_7B_INSTRUCT: "Qwen/Qwen2.5-7B-Instruct",
}

HALFVEC_EMBEDDING_MODELS: frozenset[Model] = frozenset(
    {
        Model.TEXT_EMBEDDING_3_LARGE,
        Model.GEMINI_EMBEDDING_001,
    },
)

MODEL_DISTANCE_THRESHOLDS: dict[Model, float] = {
    Model.BGE_M3: 0.4,
    Model.BGE_M3_LOCAL: 0.4,
    Model.MULTILINGUAL_E5_LARGE: 0.45,
    Model.TEXT_EMBEDDING_3_LARGE: 0.35,
    Model.GEMINI_EMBEDDING_001: 0.35,
    Model.LLAMA_3_3_70B: 0.5,
}
