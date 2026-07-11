import logging
from asyncio import to_thread
from typing import overload

from app.llms.bge_m3 import get_bge_m3_embeddings
from app.llms.models import Model

logger = logging.getLogger(__name__)

embedders = {
    Model.BGE_M3: get_bge_m3_embeddings,
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
    Generate BGE-M3 embeddings without blocking the event loop.
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

    embedder = embedders[model]

    def _call() -> list[float] | list[list[float]]:
        return embedder(texts)

    return await to_thread(_call)
