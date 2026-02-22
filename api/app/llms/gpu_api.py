import asyncio
import logging
from collections.abc import AsyncGenerator

import httpx
from fastapi.responses import StreamingResponse

from app.llms.models import GPU_API_MODELS, Model
from app.utils.http_client import get_http_client
from app.utils.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()

_STREAMING_TIMEOUT = httpx.Timeout(timeout=300.0)
_SSE_ERROR_MESSAGE = (
    "data: An error occurred while processing your request. Please try again.\n\n"
)


async def generate_gpu_api_embeddings(
    text: str | list[str],
    model: Model,
) -> list[float] | list[list[float]]:
    """
    Generate embeddings using the GPU API service.
    """
    logger.info(
        "Generating GPU API embeddings for text with length '%s' with model: %s",
        len(text) if isinstance(text, str) else sum(len(t) for t in text),
        model.value,
    )

    gpu_api_url = f"{settings.GPU_API_URL}/embeddings/embed"

    payload = {
        "input": text,
        "embeddings_model": GPU_API_MODELS[model],
    }

    client = get_http_client()
    response = await client.post(
        gpu_api_url,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=300,
    )

    response.raise_for_status()

    result = response.json()
    embeddings = result.get("embeddings")

    if embeddings is None:
        raise ValueError(f"GPU API response missing 'embeddings' key: {result}")

    return embeddings


def stream_gpu_api_response(
    user_prompt: str,
    model: Model,
    *,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> StreamingResponse:
    """
    Stream a response from the GPU API service.
    Reuses the shared HTTP client with a per-request streaming timeout.
    """
    logger.info(
        "Streaming GPU API response for user prompt length: '%d' with model: %s",
        len(user_prompt),
        model.value,
    )

    gpu_api_url = f"{settings.GPU_API_URL}/stream/"

    payload = {
        "prompt": user_prompt,
        "inference_model": model.value,
        "system_prompt": system_prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    async def stream_from_gpu_api() -> AsyncGenerator[str]:
        client = get_http_client()
        try:
            async with client.stream(
                "POST",
                gpu_api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=_STREAMING_TIMEOUT,
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    logger.error(
                        "GPU API returned error status %d: %s",
                        response.status_code,
                        error_text.decode(),
                    )
                    yield _SSE_ERROR_MESSAGE
                    return

                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk.decode("utf-8")

        except httpx.RequestError:
            logger.exception("Connection error to GPU API")
            yield _SSE_ERROR_MESSAGE
        except asyncio.CancelledError:
            logger.exception("Streaming cancelled from GPU API")

            raise
        except Exception:
            logger.exception("Unexpected error while streaming from GPU API")
            yield _SSE_ERROR_MESSAGE

    return StreamingResponse(
        stream_from_gpu_api(),
        media_type="text/event-stream",
    )
