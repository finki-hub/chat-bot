import logging
from collections.abc import Callable

from app.llms.bge_m3 import get_bge_m3_embeddings
from app.llms.inference import Lane, LaneRun, run_in_lane
from app.llms.models import Model

logger = logging.getLogger(__name__)

type Embeddings = list[float] | list[list[float]]

embedders: dict[Model, Callable[[str | list[str]], Embeddings]] = {
    Model.BGE_M3: get_bge_m3_embeddings,
}


async def generate_embeddings(
    texts: str | list[str],
    model: Model,
) -> LaneRun[Embeddings]:
    """
    Generate BGE-M3 embeddings on the embed lane without blocking the event loop.
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

    return await run_in_lane(Lane.EMBED, embedders[model], texts)
