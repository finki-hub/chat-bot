from uuid import UUID

from app.data.embedding_lifecycle import (
    BGE_M3_DIMENSIONS,
    EmbeddingBatch,
    EmbeddingCandidate,
    EmbeddingCorpus,
    canonical_chunk_text,
    canonical_diploma_text,
    canonical_professor_document_text,
    canonical_question_text,
    validate_embedding_batch,
)
from app.llms.models import (
    BGE_M3_EMBEDDING_SPEC_VERSION,
    Model,
    is_bge_m3_lifecycle_model,
)


def test_bge_m3_lifecycle_contract_includes_both_bge_aliases() -> None:
    # Given: both persisted BGE-M3 model aliases and an unrelated embedding model.
    # When: lifecycle membership and the stable version are inspected.
    # Then: only the two BGE aliases share the versioned lifecycle.
    assert BGE_M3_EMBEDDING_SPEC_VERSION == "bge-m3-v1"
    assert is_bge_m3_lifecycle_model(Model.BGE_M3)
    assert is_bge_m3_lifecycle_model(Model.BGE_M3_LOCAL)
    assert not is_bge_m3_lifecycle_model(Model.TEXT_EMBEDDING_3_LARGE)


def test_canonical_text_matches_existing_corpus_semantics() -> None:
    # Given: canonical source fields for each BGE-M3 corpus.
    # When: lifecycle text builders prepare embedding inputs.
    # Then: text matches the existing fill paths exactly.
    assert (
        canonical_question_text("Прашање", "Одговор")
        == "Наслов: Прашање\nСодржина: Одговор"
    )
    assert canonical_chunk_text("Документ", "Дел", "Текст") == (
        "Наслов: Документ (Дел)\nСодржина: Текст"
    )
    assert canonical_chunk_text("Документ", None, "Текст") == (
        "Наслов: Документ\nСодржина: Текст"
    )
    assert canonical_diploma_text(" Наслов ", " Опис ") == "Наслов\nОпис"
    assert canonical_professor_document_text(" Наслов ", None) == "Наслов"


def test_validate_embedding_batch_rejects_cardinality_and_dimension_mismatches() -> (
    None
):
    # Given: two captured dirty rows from one corpus.
    candidates = (
        EmbeddingCandidate(
            corpus=EmbeddingCorpus.QUESTION,
            id=UUID("00000000-0000-0000-0000-000000000001"),
            text="one",
            revision=1,
        ),
        EmbeddingCandidate(
            corpus=EmbeddingCorpus.QUESTION,
            id=UUID("00000000-0000-0000-0000-000000000002"),
            text="two",
            revision=1,
        ),
    )
    valid = EmbeddingBatch(
        corpus=EmbeddingCorpus.QUESTION,
        candidates=candidates,
        vectors=(
            (0.0,) * BGE_M3_DIMENSIONS,
            (1.0,) * BGE_M3_DIMENSIONS,
        ),
    )
    missing_vector = EmbeddingBatch(
        corpus=EmbeddingCorpus.QUESTION,
        candidates=candidates,
        vectors=((0.0,) * BGE_M3_DIMENSIONS,),
    )
    short_vector = EmbeddingBatch(
        corpus=EmbeddingCorpus.QUESTION,
        candidates=candidates,
        vectors=(
            (0.0,) * BGE_M3_DIMENSIONS,
            (1.0,) * (BGE_M3_DIMENSIONS - 1),
        ),
    )

    # When: provider batches are validated before persistence.
    # Then: only a complete 1024-dimensional batch is eligible for writes.
    assert validate_embedding_batch(valid)
    assert not validate_embedding_batch(missing_vector)
    assert not validate_embedding_batch(short_vector)
