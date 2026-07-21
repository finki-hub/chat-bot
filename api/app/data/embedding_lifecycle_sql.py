from enum import StrEnum
from typing import Final


class EmbeddingCorpus(StrEnum):
    QUESTION = "question"
    CHUNK = "chunk"
    DIPLOMA = "diploma"
    PROFESSOR_DOCUMENT = "professor_document"


DIRTY_SELECT_SQL: Final[dict[EmbeddingCorpus, str]] = {
    EmbeddingCorpus.QUESTION: """
        SELECT id, name, content, embedding_revision
        FROM question
        WHERE embedding_bge_m3 IS NULL
           OR embedding_bge_m3_version IS DISTINCT FROM $1
        ORDER BY id
        LIMIT $2
    """,
    EmbeddingCorpus.CHUNK: """
        SELECT c.id, d.title AS document_title, c.section, c.content,
               c.embedding_revision
        FROM chunk c
        JOIN document d ON d.id = c.document_id
        WHERE c.embedding_bge_m3 IS NULL
           OR c.embedding_bge_m3_version IS DISTINCT FROM $1
        ORDER BY c.id
        LIMIT $2
    """,
    EmbeddingCorpus.DIPLOMA: """
        SELECT id, title, description, embedding_revision
        FROM diploma
        WHERE embedding_bge_m3 IS NULL
           OR embedding_bge_m3_version IS DISTINCT FROM $1
        ORDER BY id
        LIMIT $2
    """,
    EmbeddingCorpus.PROFESSOR_DOCUMENT: """
        SELECT id, title, abstract, embedding_revision
        FROM professor_document
        WHERE embedding_bge_m3 IS NULL
           OR embedding_bge_m3_version IS DISTINCT FROM $1
        ORDER BY id
        LIMIT $2
    """,
}

COUNT_SQL: Final[dict[EmbeddingCorpus, str]] = {
    EmbeddingCorpus.QUESTION: """
        SELECT
            COUNT(*) FILTER (
                WHERE embedding_bge_m3 IS NOT NULL
                  AND embedding_bge_m3_version = $1
            ) AS ready,
            COUNT(*) FILTER (
                WHERE embedding_bge_m3 IS NULL
                   OR embedding_bge_m3_version IS DISTINCT FROM $1
            ) AS dirty
        FROM question
    """,
    EmbeddingCorpus.CHUNK: """
        SELECT
            COUNT(*) FILTER (
                WHERE embedding_bge_m3 IS NOT NULL
                  AND embedding_bge_m3_version = $1
            ) AS ready,
            COUNT(*) FILTER (
                WHERE embedding_bge_m3 IS NULL
                   OR embedding_bge_m3_version IS DISTINCT FROM $1
            ) AS dirty
        FROM chunk
    """,
    EmbeddingCorpus.DIPLOMA: """
        SELECT
            COUNT(*) FILTER (
                WHERE embedding_bge_m3 IS NOT NULL
                  AND embedding_bge_m3_version = $1
            ) AS ready,
            COUNT(*) FILTER (
                WHERE embedding_bge_m3 IS NULL
                   OR embedding_bge_m3_version IS DISTINCT FROM $1
            ) AS dirty
        FROM diploma
    """,
    EmbeddingCorpus.PROFESSOR_DOCUMENT: """
        SELECT
            COUNT(*) FILTER (
                WHERE embedding_bge_m3 IS NOT NULL
                  AND embedding_bge_m3_version = $1
            ) AS ready,
            COUNT(*) FILTER (
                WHERE embedding_bge_m3 IS NULL
                   OR embedding_bge_m3_version IS DISTINCT FROM $1
            ) AS dirty
        FROM professor_document
    """,
}

PERSIST_SQL: Final[dict[EmbeddingCorpus, str]] = {
    EmbeddingCorpus.QUESTION: """
        UPDATE question
        SET embedding_bge_m3 = $1::vector,
            embedding_bge_m3_version = $2,
            embedding_bge_m3_updated_at = NOW()
        WHERE id = $3
          AND embedding_revision = $4
        RETURNING TRUE
    """,
    EmbeddingCorpus.CHUNK: """
        UPDATE chunk
        SET embedding_bge_m3 = $1::vector,
            embedding_bge_m3_version = $2,
            embedding_bge_m3_updated_at = NOW()
        WHERE id = $3
          AND embedding_revision = $4
        RETURNING TRUE
    """,
    EmbeddingCorpus.DIPLOMA: """
        UPDATE diploma
        SET embedding_bge_m3 = $1::vector,
            embedding_bge_m3_version = $2,
            embedding_bge_m3_updated_at = NOW()
        WHERE id = $3
          AND embedding_revision = $4
        RETURNING TRUE
    """,
    EmbeddingCorpus.PROFESSOR_DOCUMENT: """
        UPDATE professor_document
        SET embedding_bge_m3 = $1::vector,
            embedding_bge_m3_version = $2,
            embedding_bge_m3_updated_at = NOW()
        WHERE id = $3
          AND embedding_revision = $4
        RETURNING TRUE
    """,
}

REBUILD_SQL: Final[dict[EmbeddingCorpus, str]] = {
    EmbeddingCorpus.QUESTION: """
        UPDATE question
        SET embedding_revision = embedding_revision + 1,
            embedding_bge_m3 = NULL,
            embedding_bge_m3_version = NULL,
            embedding_bge_m3_updated_at = NULL
    """,
    EmbeddingCorpus.CHUNK: """
        UPDATE chunk
        SET embedding_revision = embedding_revision + 1,
            embedding_bge_m3 = NULL,
            embedding_bge_m3_version = NULL,
            embedding_bge_m3_updated_at = NULL
    """,
    EmbeddingCorpus.DIPLOMA: """
        UPDATE diploma
        SET embedding_revision = embedding_revision + 1,
            embedding_bge_m3 = NULL,
            embedding_bge_m3_version = NULL,
            embedding_bge_m3_updated_at = NULL
    """,
    EmbeddingCorpus.PROFESSOR_DOCUMENT: """
        UPDATE professor_document
        SET embedding_revision = embedding_revision + 1,
            embedding_bge_m3 = NULL,
            embedding_bge_m3_version = NULL,
            embedding_bge_m3_updated_at = NULL
    """,
}
