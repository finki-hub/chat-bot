import logging

from app.llms.google import transform_query_with_google
from app.llms.models import Model
from app.llms.ollama import transform_query_with_ollama
from app.llms.openai import transform_query_with_openai
from app.llms.prompts import DEFAULT_QUERY_TRANSFORM_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


async def transform_query(
    query: str,
    model: Model,
    *,
    system_prompt: str = DEFAULT_QUERY_TRANSFORM_SYSTEM_PROMPT,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    logger.info(
        "Transforming query: '%s'",
        query,
    )

    match model:
        case (
            Model.GPT_4O_MINI
            | Model.GPT_4_1
            | Model.GPT_4_1_MINI
            | Model.GPT_4_1_NANO
            | Model.GPT_5_4_MINI
            | Model.GPT_5_2
            | Model.GPT_5_MINI
            | Model.GPT_5_NANO
        ):
            return await transform_query_with_openai(
                query,
                model,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

        case (
            Model.GEMINI_2_5_FLASH
            | Model.GEMINI_2_5_PRO
            | Model.GEMINI_3_FLASH_PREVIEW
        ):
            return await transform_query_with_google(
                query,
                model,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

        case (
            Model.LLAMA_3_3_70B
            | Model.MISTRAL
            | Model.DEEPSEEK_R1_70B
            | Model.QWEN2_5_72B
            | Model.DOMESTIC_YAK_8B_INSTRUCT_GGUF
            | Model.VEZILKALLM_GGUF
        ):
            return await transform_query_with_ollama(
                query,
                model,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

        case _:
            raise ValueError(f"Unsupported model for query transform: {model}")
