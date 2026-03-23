"""Shared text-processing helpers for the embedding pipeline."""

from app.llms.models import Model


def _add_e5_prefix(text: str, *, is_document: bool) -> str:
    """Prepend the E5 prefix required by multilingual-e5-large."""
    prefix = "passage: " if is_document else "query: "
    return f"{prefix}{text}"


def _prepare_text_for_embedding(text: str, model: Model, *, is_document: bool) -> str:
    """Apply any model-specific text transformations before embedding."""
    if model == Model.MULTILINGUAL_E5_LARGE:
        return _add_e5_prefix(text, is_document=is_document)
    return text
