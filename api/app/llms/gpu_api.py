import logging
from typing import Annotated

import httpx
from pydantic import Field, TypeAdapter

from app.llms.models import GPU_API_MODELS, MODEL_EMBEDDING_DIMENSIONS, Model
from app.utils.http_client import get_http_client
from app.utils.settings import Settings
from app.utils.timing import current_distinct_id, current_response_id

logger = logging.getLogger(__name__)

settings = Settings()

_EMBEDDINGS_TIMEOUT = httpx.Timeout(timeout=60.0)
type EmbeddingValue = Annotated[float, Field(strict=True, allow_inf_nan=False)]
type Embedding = list[EmbeddingValue]
_EMBEDDING_ADAPTER: TypeAdapter[Embedding] = TypeAdapter(Embedding)
_EMBEDDING_BATCH_ADAPTER: TypeAdapter[list[Embedding]] = TypeAdapter(list[Embedding])


def _forwarded_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    rid = current_response_id()
    if rid:
        headers["X-Response-Id"] = rid
    did = current_distinct_id()
    if did:
        headers["X-Distinct-Id"] = did
    return headers


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
        headers={"Content-Type": "application/json", **_forwarded_headers()},
        timeout=_EMBEDDINGS_TIMEOUT,
    )

    response.raise_for_status()

    result = response.json()
    embeddings = result.get("embeddings")

    if embeddings is None:
        raise ValueError(f"GPU API response missing 'embeddings' key: {result}")

    expected_dimensions = MODEL_EMBEDDING_DIMENSIONS[model]
    if isinstance(text, str):
        embedding = _EMBEDDING_ADAPTER.validate_python(embeddings)
        if len(embedding) != expected_dimensions:
            msg = (
                "GPU API returned an embedding with "
                f"{len(embedding)} dimensions; expected {expected_dimensions}"
            )
            raise ValueError(msg)
        return embedding

    embedding_batch = _EMBEDDING_BATCH_ADAPTER.validate_python(embeddings)
    if len(embedding_batch) != len(text):
        msg = (
            f"GPU API returned {len(embedding_batch)} embeddings for {len(text)} inputs"
        )
        raise ValueError(msg)
    for embedding in embedding_batch:
        if len(embedding) != expected_dimensions:
            msg = (
                "GPU API returned an embedding with "
                f"{len(embedding)} dimensions; expected {expected_dimensions}"
            )
            raise ValueError(msg)
    return embedding_batch
