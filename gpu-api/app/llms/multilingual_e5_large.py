import logging
from typing import overload

import torch
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

_multilingual_e5_large_embedder: HuggingFaceEmbeddings | None = None


@overload
def get_multilingual_e5_large_embeddings(text: str) -> list[float]: ...


@overload
def get_multilingual_e5_large_embeddings(text: list[str]) -> list[list[float]]: ...


def get_multilingual_e5_large_embeddings(
    text: str | list[str],
) -> list[float] | list[list[float]]:
    """
    Get embeddings using the intfloat/multilingual-e5-large model from Hugging Face.
    """
    global _multilingual_e5_large_embedder  # noqa: PLW0603

    if _multilingual_e5_large_embedder is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading Multilingual E5 Large embeddings model on %s", device)
        _multilingual_e5_large_embedder = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

    if isinstance(text, str):
        return _multilingual_e5_large_embedder.embed_query(text)

    return _multilingual_e5_large_embedder.embed_documents(text)
