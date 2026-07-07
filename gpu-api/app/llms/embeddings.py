import asyncio
import logging
from typing import overload

from fastapi import HTTPException, status

from app.llms.bge_m3 import get_bge_m3_embeddings
from app.llms.models import Model
from app.llms.multilingual_e5_large import get_multilingual_e5_large_embeddings

logger = logging.getLogger(__name__)

embedders = {
    Model.BGE_M3: get_bge_m3_embeddings,
    Model.MULTILINGUAL_E5_LARGE: get_multilingual_e5_large_embeddings,
}


@overload
async def generate_embeddings(
    texts: str,
    model: Model,
) -> list[float]: ...


@overload
async def generate_embeddings(
    texts: list[str],
    model: Model,
) -> list[list[float]]: ...


async def generate_embeddings(
    texts: str | list[str],
    model: Model,
) -> list[float] | list[list[float]]:
    """
    Dispatch to the appropriate embedder, offloading blocking calls.
    Raises HTTPException(400) if the model isn't supported.
    """
    input_chars = (
        len(texts) if isinstance(texts, str) else sum(len(text) for text in texts)
    )
    input_count = 1 if isinstance(texts, str) else len(texts)
    logger.info(
        "Generating embeddings: model=%s input_chars=%d count=%d",
        model.value,
        input_chars,
        input_count,
    )

    embedder = embedders.get(model)

    if embedder is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model {model.value} is not supported for embeddings.",
        )

    def _call() -> list[float] | list[list[float]]:
        return embedder(texts)

    return await asyncio.to_thread(_call)
