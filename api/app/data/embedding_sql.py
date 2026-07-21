from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from app.llms.models import (
    BGE_M3_EMBEDDING_SPEC_VERSION,
    HALFVEC_EMBEDDING_MODELS,
    MODEL_EMBEDDINGS_COLUMNS,
    Model,
    is_bge_m3_lifecycle_model,
)

type SqlTableAlias = Literal["c"]


@dataclass(frozen=True, slots=True)
class EmbeddingVectorSql:
    """Safe SQL identifier fragments for a known embedding model column."""

    column: str
    column_ref: str
    distance_operand: str
    query_operand: str


@dataclass(frozen=True, slots=True)
class EmbeddingPredicate:
    """One allowlisted embedding predicate and its bound values."""

    sql: str
    parameters: tuple[str, ...]


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


def dirty_embedding_predicate(
    model: Model,
    column_ref: str,
    *,
    version_parameter: int = 1,
) -> EmbeddingPredicate:
    """Return the model-specific predicate for rows needing an embedding fill."""
    if is_bge_m3_lifecycle_model(model):
        version_ref = (
            column_ref.removesuffix("embedding_bge_m3") + "embedding_bge_m3_version"
        )
        return EmbeddingPredicate(
            sql=(
                f"({column_ref} IS NULL OR {version_ref} IS DISTINCT FROM "
                f"${version_parameter})"
            ),
            parameters=(BGE_M3_EMBEDDING_SPEC_VERSION,),
        )
    return EmbeddingPredicate(sql=f"{column_ref} IS NULL", parameters=())


def current_embedding_predicate(
    model: Model,
    column_ref: str,
    *,
    version_parameter: int,
) -> EmbeddingPredicate:
    """Return the model-specific predicate for rows eligible for retrieval."""
    if is_bge_m3_lifecycle_model(model):
        version_ref = (
            column_ref.removesuffix("embedding_bge_m3") + "embedding_bge_m3_version"
        )
        return EmbeddingPredicate(
            sql=f"{column_ref} IS NOT NULL AND {version_ref} = ${version_parameter}",
            parameters=(BGE_M3_EMBEDDING_SPEC_VERSION,),
        )
    return EmbeddingPredicate(sql=f"{column_ref} IS NOT NULL", parameters=())
