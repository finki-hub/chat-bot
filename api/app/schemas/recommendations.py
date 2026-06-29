from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


class RecommendationRequestSchema(BaseModel):
    title: str = Field(
        min_length=1,
        examples=["Систем за препорака на ментори базиран на вградувања"],
        description=(
            "Thesis title (Macedonian/Cyrillic). Required. Short titles work, "
            "but optional abstract/keywords/area fields improve retrieval."
        ),
    )
    abstract: str | None = Field(
        default=None,
        description="Optional thesis abstract or longer proposal text used only for retrieval.",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Optional topical keywords used to enrich the retrieval query.",
    )
    study_program: str | None = Field(
        default=None,
        description="Optional study program/context used to enrich the retrieval query.",
    )
    research_area: str | None = Field(
        default=None,
        description="Optional broad research area used to enrich the retrieval query.",
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
    exclude_professors: list[str] = Field(
        default_factory=list,
        description="Canonical professor names that must not appear in the committee.",
    )
    include_professors: list[str] = Field(
        default_factory=list,
        description="Canonical professor names that must appear in the committee if feasible.",
    )
    alternatives: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of committee alternatives to return, including the selected one.",
    )

    @field_validator("title", "mentor", "abstract", "study_program", "research_area")
    @classmethod
    def _strip(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank or whitespace-only")
        return stripped

    @field_validator("keywords", "exclude_professors", "include_professors")
    @classmethod
    def _strip_list(cls, value: list[str]) -> list[str]:
        seen: dict[str, None] = {}
        for item in value:
            stripped = item.strip()
            if stripped:
                seen.setdefault(stripped, None)
        return list(seen)

    @model_validator(mode="after")
    def _validate_constraints(self) -> RecommendationRequestSchema:
        overlap = set(self.exclude_professors) & set(self.include_professors)
        if overlap:
            raise ValueError(
                "include_professors and exclude_professors must not overlap",
            )
        if self.mentor is not None and self.mentor in self.exclude_professors:
            raise ValueError("mentor must not be listed in exclude_professors")
        return self


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
    supporting_paper_titles: list[str] = Field(
        default_factory=list,
        description="Human-readable paper evidence that supports this person's expertise.",
    )


class SupportingDiplomaSchema(BaseModel):
    id: UUID = Field(description="Diploma id used as historical evidence")
    external_id: str = Field(description="Source external id for the defended thesis")
    title: str = Field(description="Historical defended thesis title")
    mentor: str = Field(description="Historical defense mentor")
    members: list[str] = Field(description="Historical defense committee members")
    similarity: float = Field(description="Vector similarity to the requested proposal")
    rerank_score: float | None = Field(
        default=None,
        description="Cross-encoder rerank score, if reranking succeeded",
    )


class CommitteeAlternativeSchema(BaseModel):
    mentor: str | None = Field(description="Alternative committee mentor")
    members: list[str] = Field(description="Alternative unordered member pair")
    score: float = Field(description="Internal committee objective score")
    supporting_diploma_ids: list[UUID] = Field(
        default_factory=list,
        description="Diploma ids supporting this alternative",
    )


class RecommendationEvidenceSchema(BaseModel):
    similar_diplomas: list[SupportingDiplomaSchema] = Field(
        default_factory=list,
        description="Top similar defended theses used as evidence.",
    )
    supporting_paper_titles: list[str] = Field(
        default_factory=list,
        description="Paper titles supporting selected committee expertise.",
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
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Heuristic confidence in the recommendation evidence strength.",
    )
    confidence_reasons: list[str] = Field(
        default_factory=list,
        description="Short machine-readable reasons behind the confidence score.",
    )
    evidence: RecommendationEvidenceSchema = Field(
        default_factory=RecommendationEvidenceSchema,
        description="Human-readable evidence for the selected committee.",
    )
    alternatives: list[CommitteeAlternativeSchema] = Field(
        default_factory=list,
        description="Ranked committee alternatives, including the selected committee first.",
    )
