import logging

import torch
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.model import PairInput

from app.utils.exceptions import ModelNotReadyError

logger = logging.getLogger(__name__)

_reranker_model: CrossEncoder | None = None


def init_reranker(model_name: str) -> None:
    """
    Initialize the cross-encoder reranker during application startup.
    Called from the lifespan manager. The default (bge-reranker-v2-m3) is
    multilingual; the v1 "large" was tuned for Chinese/English only.
    """
    global _reranker_model  # noqa: PLW0603

    logger.info("Initializing reranker model: %s", model_name)

    if _reranker_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load in fp16 on CUDA (Ampere+): halves VRAM and cuts rerank latency
        # with negligible ranking-quality impact. CPU stays fp32 (no fast fp16).
        model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}

        _reranker_model = CrossEncoder(
            model_name,
            device=device,
            model_kwargs=model_kwargs,
        )

        logger.info("Reranker model initialized successfully on device: %s", device)


def rerank_documents(query: str, documents: list[str]) -> list[tuple[float, int]]:
    """
    Re-ranks a list of documents based on their relevance to a query
    using the pre-loaded cross-encoder model.

    Returns a list of (score, original_index) tuples sorted by score descending.
    """
    logger.info(
        "Reranking %d documents for query: %s",
        len(documents),
        query,
    )

    if not documents or not query:
        return [(0.0, idx) for idx in range(len(documents))]
    if _reranker_model is None:
        raise ModelNotReadyError

    model_inputs: list[PairInput] = [(query, doc) for doc in documents]

    scores = _reranker_model.predict(model_inputs)

    if logger.isEnabledFor(logging.DEBUG):
        score_values = [float(s) for s in scores]
        logger.debug(
            "Rerank score distribution: min=%.4f max=%.4f mean=%.4f (n=%d)",
            min(score_values),
            max(score_values),
            sum(score_values) / len(score_values),
            len(score_values),
        )

    return sorted(
        ((float(score), idx) for idx, score in enumerate(scores)),
        key=lambda x: x[0],
        reverse=True,
    )
