from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from app.llms.models import HALFVEC_EMBEDDING_MODELS, MODEL_EMBEDDINGS_COLUMNS, Model

type SqlTableAlias = Literal["c"]


@dataclass(frozen=True, slots=True)
class EmbeddingVectorSql:
    """Safe SQL identifier fragments for a known embedding model column."""

    column: str
    column_ref: str
    distance_operand: str
    query_operand: str


def embedding_column_name(model: Model) -> str:
    """Return the allowlisted database column for an embedding model."""
    return MODEL_EMBEDDINGS_COLUMNS[model]


def embedding_vector_sql(
    model: Model,
    embedded_query: Sequence[float],
    *,
    table_alias: SqlTableAlias | None = None,
) -> EmbeddingVectorSql:
    """Build pgvector/halfvec SQL fragments from allowlisted model metadata."""
    column = embedding_column_name(model)
    column_ref = f"{table_alias}.{column}" if table_alias else column

    if model not in HALFVEC_EMBEDDING_MODELS:
        return EmbeddingVectorSql(
            column=column,
            column_ref=column_ref,
            distance_operand=column_ref,
            query_operand="$1",
        )

    dims = len(embedded_query)
    return EmbeddingVectorSql(
        column=column,
        column_ref=column_ref,
        distance_operand=f"{column_ref}::halfvec({dims})",
        query_operand=f"$1::halfvec({dims})",
    )
