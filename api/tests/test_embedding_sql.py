from app.data.embedding_sql import embedding_column_name, embedding_vector_sql
from app.llms.models import Model


def test_embedding_column_name_returns_allowlisted_model_column() -> None:
    assert embedding_column_name(Model.BGE_M3) == "embedding_bge_m3"


def test_embedding_vector_sql_uses_plain_vector_for_low_dimension_model() -> None:
    sql = embedding_vector_sql(Model.BGE_M3, [0.1, 0.2])

    assert sql.column == "embedding_bge_m3"
    assert sql.column_ref == "embedding_bge_m3"
    assert sql.distance_operand == "embedding_bge_m3"
    assert sql.query_operand == "$1"


def test_embedding_vector_sql_uses_halfvec_with_alias_for_high_dimension_model() -> (
    None
):
    sql = embedding_vector_sql(
        Model.TEXT_EMBEDDING_3_LARGE,
        [0.1, 0.2, 0.3],
        table_alias="c",
    )

    assert sql.column == "embedding_text_embedding_3_large"
    assert sql.column_ref == "c.embedding_text_embedding_3_large"
    assert sql.distance_operand == "c.embedding_text_embedding_3_large::halfvec(3)"
    assert sql.query_operand == "$1::halfvec(3)"
