from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ProfessorSchema"]


class ProfessorSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int | None = Field(
        default=None,
        description="Directus staff id, when available.",
    )
    name: str = Field(
        examples=["Александар Стојменски"],
        description="Canonical staff name, matching the names used by the recommender.",
    )
    title: str | None = Field(
        default=None,
        examples=["д-р"],
        description="Academic title.",
    )
    position: str | None = Field(
        default=None,
        examples=["Доцент"],
        description="Current position at the faculty.",
    )
    email: str | None = Field(
        default=None,
        description="Public faculty email address.",
    )
    cabinet: str | None = Field(
        default=None,
        description="Office/cabinet location.",
    )
    profile: str | None = Field(
        default=None,
        description="Public faculty profile page.",
    )
    courses: str | None = Field(
        default=None,
        description="Public courses profile page.",
    )
    consultations: str | None = Field(
        default=None,
        description="Public consultations page.",
    )
    photo: str | None = Field(
        default=None,
        description="Public staff photo URL.",
    )
    mentor_eligible: bool = Field(
        description=(
            "Whether this staff member holds a teaching-scientific title that "
            "allows them to mentor a diploma thesis."
        ),
    )
