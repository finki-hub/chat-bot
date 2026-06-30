from datetime import date, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from app.constants.defaults import DEFAULT_EMBEDDINGS_MODEL
from app.llms.models import Model
from app.schemas.recommendations import (
    CommitteeAlternativeSchema,
    PersonScoreSchema,
    RecommendationEvidenceSchema,
    RecommendationRequestSchema,
    RecommendationResponseSchema,
    SupportingDiplomaSchema,
)

__all__ = [
    "CommitteeAlternativeSchema",
    "DiplomaSchema",
    "FillDiplomaEmbeddingsSchema",
    "PersonScoreSchema",
    "ProfessorGroupSchema",
    "RecommendationEvidenceSchema",
    "RecommendationRequestSchema",
    "RecommendationResponseSchema",
    "SupportingDiplomaSchema",
]


class DiplomaSchema(BaseModel):
    id: UUID = Field(
        examples=["3fa85f64-5717-4562-b3fc-2c963f66afa6"],
        description="Unique identifier for the diploma defense",
    )
    external_id: str = Field(
        examples=["a1b2c3d4e5f6"],
        description="Content-hash idempotency key (sha256 of fileId|student|title|date)",
    )
    title: str = Field(
        examples=["Систем за препорака на ментори базиран на вградувања"],
        description="Thesis title (Macedonian/Cyrillic)",
    )
    description: str = Field(
        description="Thesis description (corpus-only; never on the request schema)",
    )
    mentor: str = Field(
        examples=["Соња Гиевска"],
        description="Canonical mentor name (source of truth)",
    )
    member1: str | None = Field(
        default=None,
        examples=["Иван Чорбев"],
        description="Canonical name of the first committee member",
    )
    member2: str | None = Field(
        default=None,
        examples=["Боро Јакимовски"],
        description="Canonical name of the second committee member",
    )
    status: str = Field(
        examples=["Одбрана"],
        description="Defense status",
    )
    date_of_submission: date | None = Field(
        default=None,
        examples=["2024-09-15"],
        description="Submission date, parsed from 'DD.MM.YYYY'",
    )
    created_at: datetime = Field(examples=["2025-06-05T14:48:00Z"])
    updated_at: datetime = Field(examples=["2025-06-06T09:24:00Z"])
    distance: float | None = Field(
        default=None,
        examples=[0.123456],
        description="Distance metric for similarity search, if applicable",
    )


class FillDiplomaEmbeddingsSchema(BaseModel):
    embeddings_model: Model = Field(
        default=DEFAULT_EMBEDDINGS_MODEL,
        examples=[DEFAULT_EMBEDDINGS_MODEL.value],
        description="Which embedding model to use",
    )
    all_models: bool = Field(
        default=False,
        description="Whether to regenerate embeddings for all models or just the specified one",
    )


class ProfessorGroupSchema(BaseModel):
    source: Literal["defense", "coauthor"] = Field(
        examples=["defense"],
        description="'defense' (committee co-occurrence) or 'coauthor' (paper co-authorship)",
    )
    window_start: int = Field(
        examples=[2023],
        description="Window start year (inclusive)",
    )
    window_end: int = Field(examples=[2025], description="Window end year (inclusive)")
    group_index: int = Field(
        examples=[0],
        description="Group rank within the window (0 = largest)",
    )
    members: list[str] = Field(
        description="Canonical professor names in this co-working cohort",
    )
    size: int = Field(examples=[12], description="Number of members")
    min_weight: int = Field(
        examples=[2],
        description="Min co-occurrences within the window to form an edge",
    )
