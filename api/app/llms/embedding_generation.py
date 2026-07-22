import asyncio
import logging
from typing import overload

from fastapi import HTTPException, status

from app.llms.google import generate_google_embeddings
from app.llms.gpu_api import generate_gpu_api_embeddings
from app.llms.models import ACTIVE_EMBEDDING_MODELS, ALL_MODELS_EMBEDDINGS, Model
from app.llms.openai import generate_openai_embeddings
from app.llms.provider_credentials import (
    LlmProviderCredentials,
    ProviderCredentialRequiredError,
    provider_for_model,
)

logger = logging.getLogger(__name__)
EMBEDDING_MAX_ATTEMPTS = 3
_EMBEDDING_RETRY_BASE_DELAY = 0.5


async def _dispatch_embeddings(
    text: str | list[str],
    model: Model,
    *,
    is_document: bool,
    credentials: LlmProviderCredentials | None,
) -> list[float] | list[list[float]]:
    embeddings: list[float] | list[list[float]]
    match model:
        case Model.BGE_M3_LOCAL:
            embeddings = await generate_gpu_api_embeddings(text, model)
        case Model.TEXT_EMBEDDING_3_LARGE:
            embeddings = await generate_openai_embeddings(
                text,
                model,
                None if credentials is None else credentials.openai,
            )
        case Model.GEMINI_EMBEDDING_001:
            embeddings = await generate_google_embeddings(
                text,
                model,
                is_document=is_document,
                credential=None if credentials is None else credentials.google,
            )
        case _:
            raise ValueError(f"Unsupported model: {model}")
    return embeddings


@overload
async def generate_embeddings(
    text: str,
    model: Model,
    *,
    is_document: bool = ...,
    credentials: LlmProviderCredentials | None = None,
) -> list[float]:
    raise NotImplementedError


@overload
async def generate_embeddings(
    text: list[str],
    model: Model,
    *,
    is_document: bool = ...,
    credentials: LlmProviderCredentials | None = None,
) -> list[list[float]]:
    raise NotImplementedError


async def generate_embeddings(
    text: str | list[str],
    model: Model,
    *,
    is_document: bool = False,
    credentials: LlmProviderCredentials | None = None,
) -> list[float] | list[list[float]]:
    text_count = 1 if isinstance(text, str) else len(text)
    total_chars = len(text) if isinstance(text, str) else sum(map(len, text))
    logger.info(
        "Generating embeddings model=%s text_count=%d total_chars=%d",
        model.value,
        text_count,
        total_chars,
    )
    delay = _EMBEDDING_RETRY_BASE_DELAY
    for attempt in range(1, EMBEDDING_MAX_ATTEMPTS + 1):
        try:
            return await _dispatch_embeddings(
                text,
                model,
                is_document=is_document,
                credentials=credentials,
            )
        except ProviderCredentialRequiredError:
            raise
        except ValueError:
            raise
        except Exception as exc:
            if attempt >= EMBEDDING_MAX_ATTEMPTS:
                logger.log(
                    logging.ERROR,
                    "Embedding generation failed model=%s attempts=%d error_type=%s",
                    model.value,
                    attempt,
                    type(exc).__name__,
                )
                raise
            logger.warning(
                "Embedding attempt %d/%d for model %s failed; retrying in %.1fs",
                attempt,
                EMBEDDING_MAX_ATTEMPTS,
                model.value,
                delay,
            )
            await asyncio.sleep(delay)
            delay *= 2
    raise RuntimeError("Unreachable embedding retry loop")


def ensure_self_hosted_embedding_model(model: Model) -> None:
    if model not in ACTIVE_EMBEDDING_MODELS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported embedding model: {model.value}",
        )
    if provider_for_model(model) is not None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="Hosted embeddings require a user-authenticated BYOK chat request",
        )


def resolve_fill_models(model: Model, *, all_models: bool) -> list[Model]:
    if all_models:
        return list(ALL_MODELS_EMBEDDINGS)
    ensure_self_hosted_embedding_model(model)
    return [model]
