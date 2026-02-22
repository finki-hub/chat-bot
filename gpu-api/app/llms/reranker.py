import logging

import torch
from sentence_transformers import CrossEncoder

from app.utils.exceptions import ModelNotReadyError

logger = logging.getLogger(__name__)

_reranker_model: CrossEncoder | None = None


def init_reranker() -> None:
    """
    Initializes the BGE Re-ranker model during application startup.
    This function should be called from the lifespan manager.
    """
    global _reranker_model  # noqa: PLW0603

    logger.info("Initializing reranker model...")

    if _reranker_model is None:
        model_name = "BAAI/bge-reranker-large"

        device = "cuda" if torch.cuda.is_available() else "cpu"

        _reranker_model = CrossEncoder(model_name, device=device)

        logger.info("Reranker model initialized successfully on device: %s", device)


def rerank_documents(query: str, documents: list[str]) -> list[tuple[float, str]]:
    """
    Re-ranks a list of documents based on their relevance to a query
    using the pre-loaded cross-encoder model.

    Returns a list of (score, document) tuples sorted by score descending.
    """
    logger.info(
        "Reranking %d documents for query: %s",
        len(documents),
        query,
    )

    if not documents or not query:
        return [(0.0, doc) for doc in documents]
    if _reranker_model is None:
        raise ModelNotReadyError

    model_inputs = [[query, doc] for doc in documents]

    scores = _reranker_model.predict(model_inputs)

    scored_docs = sorted(
        zip(scores, documents, strict=False),
        key=lambda x: x[0],
        reverse=True,
    )

    return [(float(score), doc) for score, doc in scored_docs]
