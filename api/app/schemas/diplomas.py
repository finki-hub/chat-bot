from datetime import date, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.constants.defaults import DEFAULT_EMBEDDINGS_MODEL
from app.llms.models import Model


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


class RecommendationRequestSchema(BaseModel):
    title: str = Field(
        min_length=1,
        examples=["Систем за препорака на ментори базиран на вградувања"],
        description=(
            "Thesis title (Macedonian/Cyrillic) — the ONLY text input. Required. "
            "Usually short, occasionally a longer line. No description field exists."
        ),
    )
    mentor: str | None = Field(
        default=None,
        examples=["Соња Гиевска"],
        description=(
            "Optional canonical mentor name. Provided -> MEMBERS-ONLY: recommend only "
            "the two members; the mentor is fixed, excluded from candidates, and used as "
            "a signal. Omitted -> FULL: recommend mentor + two members."
        ),
    )
    mentor_topk: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Mentor candidate breadth (FULL mode only).",
    )

    @field_validator("title", "mentor")
    @classmethod
    def _strip(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank or whitespace-only")
        return stripped


class PersonScoreSchema(BaseModel):
    name: str = Field(
        examples=["Соња Гиевска"],
        description="Canonical professor name",
    )
    score: float | None = Field(
        default=None,
        examples=[0.873],
        description="Blended total score (None/echo for a given mentor)",
    )
    defense_score: float = Field(
        examples=[0.62],
        description="Defense co-occurrence component (explainability/ablation)",
    )
    expertise_score: float = Field(
        examples=[0.41],
        description="Paper-expertise component (explainability/ablation)",
    )
    buddy_score: float = Field(
        default=0.0,
        examples=[0.0],
        description="Co-authorship component (0.0 when the signal is off)",
    )
    prior_score: float = Field(
        default=0.0,
        examples=[0.42],
        description=(
            "Mentor-conditioned co-membership prior component — the weighted habitual "
            "'this mentor usually sits with' contribution to this member's score."
        ),
    )
    supporting_diploma_ids: list[UUID] = Field(
        default_factory=list,
        description="Evidence: diploma ids that support this person",
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


class RecommendationResponseSchema(BaseModel):
    mode: Literal["full", "members_only"] = Field(
        examples=["full"],
        description="Inferred mode: 'full' (title only) or 'members_only' (title + mentor)",
    )
    mentor: PersonScoreSchema = Field(
        description="FULL: recommended mentor; MEMBERS-ONLY: echoed given mentor",
    )
    mentor_is_given: bool = Field(
        examples=[False],
        description="False in FULL, True in MEMBERS-ONLY",
    )
    members: list[PersonScoreSchema] = Field(
        description="The unordered member pair (2 entries); ordered by score desc for stable JSON only",
    )
    supporting_diploma_ids: list[UUID] = Field(
        default_factory=list,
        description="Top-level evidence trail",
    )
