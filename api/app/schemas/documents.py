from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from app.constants.defaults import DEFAULT_EMBEDDINGS_MODEL
from app.llms.models import Model


class DocumentSchema(BaseModel):
    id: UUID = Field(
        examples=["3fa85f64-5717-4562-b3fc-2c963f66afa6"],
        description="Unique identifier for the document",
    )
    name: str = Field(
        examples=["statut-finki-2019"],
        description="Unique key or slug of the document",
    )
    title: str = Field(
        examples=["Статут на ФИНКИ"],
        description="Human-readable document title",
    )
    source_type: str | None = Field(
        default=None,
        examples=["pdf"],
        description="Original source format: 'markdown' | 'text' | 'pdf' | 'docx'",
    )
    source_hash: str | None = Field(
        default=None,
        description="Hash of the source Markdown, used to skip unchanged re-ingests",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Arbitrary extra metadata (gazette number, dates, source URL, ...)",
    )
    user_id: str | None = Field(
        default=None,
        examples=["198249751001563136"],
        description="Discord ID of the user who ingested this document",
    )
    created_at: datetime = Field(examples=["2025-06-05T14:48:00Z"])
    updated_at: datetime = Field(examples=["2025-06-06T09:24:00Z"])
    chunk_count: int | None = Field(
        default=None,
        examples=[42],
        description="Number of chunks stored for this document",
    )


class ChunkSchema(BaseModel):
    id: UUID = Field(examples=["3fa85f64-5717-4562-b3fc-2c963f66afa6"])
    document_id: UUID = Field(examples=["3fa85f64-5717-4562-b3fc-2c963f66afa6"])
    document_name: str = Field(examples=["statut-finki-2019"])
    document_title: str = Field(examples=["Статут на ФИНКИ"])
    chunk_index: int = Field(examples=[0])
    section: str | None = Field(
        default=None,
        examples=["Член 12"],
        description="Source section label (legal article or heading)",
    )
    content: str = Field(description="The chunk text")
    distance: float | None = Field(
        default=None,
        examples=[0.123456],
        description="Distance metric for similarity search, if applicable",
    )


class IngestDocumentSchema(BaseModel):
    name: str = Field(
        examples=["statut-finki-2019"],
        description="Unique key or slug for the document",
    )
    title: str = Field(
        examples=["Статут на ФИНКИ"],
        description="Human-readable document title",
    )
    content: str = Field(
        description="The document body as normalised Markdown (членови as '# Член N')",
    )
    source_type: str | None = Field(
        default=None,
        examples=["pdf"],
        description="Original source format: 'markdown' | 'text' | 'pdf' | 'docx'",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Arbitrary extra metadata to store with the document",
    )
    user_id: str | None = Field(
        default=None,
        examples=["198249751001563136"],
    )


class FillChunkEmbeddingsSchema(BaseModel):
    embeddings_model: Model = Field(
        default=DEFAULT_EMBEDDINGS_MODEL,
        examples=[DEFAULT_EMBEDDINGS_MODEL.value],
        description="Which embedding model to use",
    )
    documents: list[str] | None = Field(
        default=None,
        examples=[["statut-finki-2019"]],
        description="Document names to (re)embed. If None, embeds chunks missing the column.",
    )
    all_chunks: bool = Field(
        default=False,
        description="Whether to regenerate _all_ chunk embeddings vs. only missing ones",
    )
    all_models: bool = Field(
        default=False,
        description="Whether to regenerate embeddings for all models or just the specified one",
    )
