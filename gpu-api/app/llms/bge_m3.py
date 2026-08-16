import logging
import threading
from typing import overload

import torch
from langchain_huggingface import HuggingFaceEmbeddings

from app.utils.exceptions import ModelNotReadyError

logger = logging.getLogger(__name__)

_bge_m3_embedder: HuggingFaceEmbeddings | None = None
_bge_m3_init_thread: threading.Thread | None = None
_bge_m3_lock = threading.Lock()


@overload
def get_bge_m3_embeddings(text: str) -> list[float]: ...


@overload
def get_bge_m3_embeddings(text: list[str]) -> list[list[float]]: ...


def get_bge_m3_embeddings(text: str | list[str]) -> list[float] | list[list[float]]:
    """
    Get embeddings using BGE M3.  If the model is not yet initialized,
    spin up initialization in background (once) and raise ModelNotReady.
    """
    global _bge_m3_init_thread, _bge_m3_embedder  # ruff: ignore[PLW0602, PLW0603] -- process-local lazy model state

    if _bge_m3_embedder is None:
        with _bge_m3_lock:
            if _bge_m3_init_thread is None or not _bge_m3_init_thread.is_alive():
                logger.info("Starting background thread to load BGE M3 embedder")
                _bge_m3_init_thread = threading.Thread(
                    target=init_bge_m3_embedder,
                    daemon=True,
                )
                _bge_m3_init_thread.start()

        raise ModelNotReadyError

    if isinstance(text, str):
        return _bge_m3_embedder.embed_query(text)

    return _bge_m3_embedder.embed_documents(text)


def init_bge_m3_embedder() -> None:
    """
    Blocking init of the BGE M3 embedder. Runs in a background thread.
    """
    global _bge_m3_embedder  # ruff: ignore[PLW0603] -- initializer owns process-local model state

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        logger.info("Loading BGE M3 model on %s", device)

        _bge_m3_embedder = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

        logger.info("BGE M3 model loaded successfully.")
    except Exception:
        logger.exception("Failed to load BGE M3 embedder.")
