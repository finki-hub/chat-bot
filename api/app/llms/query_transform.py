import logging

from app.llms.anthropic import transform_query_with_anthropic
from app.llms.google import transform_query_with_google
from app.llms.models import (
    ANTHROPIC_QUERY_TRANSFORM_MODELS,
    GOOGLE_QUERY_TRANSFORM_MODELS,
    OLLAMA_QUERY_TRANSFORM_MODELS,
    OPENAI_QUERY_TRANSFORM_MODELS,
    ChatModel,
    Model,
)
from app.llms.ollama import transform_query_with_ollama
from app.llms.openai import transform_query_with_openai
from app.llms.prompts import DEFAULT_QUERY_TRANSFORM_SYSTEM_PROMPT
from app.llms.provider_credentials import LlmProviderCredentials

logger = logging.getLogger(__name__)


async def transform_query(
    query: str,
    model: ChatModel,
    *,
    system_prompt: str = DEFAULT_QUERY_TRANSFORM_SYSTEM_PROMPT,
    temperature: float,
    top_p: float,
    max_tokens: int,
    credentials: LlmProviderCredentials | None = None,
) -> str:
    logger.info(
        "Transforming query with model=%s query_length=%d",
        model.value if isinstance(model, Model) else model,
        len(query),
    )

    if not isinstance(model, Model):
        return await transform_query_with_ollama(
            query,
            model,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            credential=None if credentials is None else credentials.ollama,
        )

    match model:
        case model if model in OPENAI_QUERY_TRANSFORM_MODELS:
            return await transform_query_with_openai(
                query,
                model,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                credential=None if credentials is None else credentials.openai,
            )

        case model if model in GOOGLE_QUERY_TRANSFORM_MODELS:
            return await transform_query_with_google(
                query,
                model,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                credential=None if credentials is None else credentials.google,
            )

        case model if model in ANTHROPIC_QUERY_TRANSFORM_MODELS:
            return await transform_query_with_anthropic(
                query,
                model,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                credential=None if credentials is None else credentials.anthropic,
            )

        case model if model in OLLAMA_QUERY_TRANSFORM_MODELS:
            return await transform_query_with_ollama(
                query,
                model,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                credential=None if credentials is None else credentials.ollama,
            )

        case _:
            raise ValueError(f"Unsupported model for query transform: {model}")
