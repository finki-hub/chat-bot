import json

from langchain_core.tools import BaseTool, StructuredTool

from app.data.connection import Database
from app.schemas.recommendations import RecommendationRequestSchema
from app.services.recommendations import (
    ActiveStaffDirectoryUnavailableError,
    InactiveStaffRequestedError,
    NoSimilarDefensesError,
    generate_committee_recommendation,
)

_RECOMMENDATION_TOOL_NAME = "recommend_diploma_committee"
_RECOMMENDATION_TOOL_DESCRIPTION = (
    "Препорачај комисија за дипломска работа на ФИНКИ од интерната база со "
    "одбранети дипломски работи и професорски публикации. Користи ја алатката "
    "кога корисникот прашува кој треба да биде ментор или членови на комисија "
    "за предложена дипломска тема. Испрати наслов и, ако се познати, апстракт, "
    "клучни зборови, студиска програма, област, фиксен ментор, вклучени "
    "професори или исклучени професори."
)


def _error_json(message: str, **extra: list[str]) -> str:
    return json.dumps({"error": message, **extra}, ensure_ascii=False)


def _inactive_staff_error(exc: InactiveStaffRequestedError) -> str:
    match exc.field:
        case "mentor":
            return _error_json("mentor must be an active staff member")
        case "include_professors":
            return _error_json(
                "include_professors must contain only active staff members",
                inactive_professors=list(exc.names),
            )
        case _:
            return _error_json("Unexpected inactive staff request")


def build_recommendation_tools(db: Database) -> list[BaseTool]:
    async def recommend_diploma_committee(
        title: str,
        abstract: str | None = None,
        keywords: list[str] | None = None,
        study_program: str | None = None,
        research_area: str | None = None,
        mentor: str | None = None,
        mentor_topk: int = 3,
        exclude_professors: list[str] | None = None,
        include_professors: list[str] | None = None,
        alternatives: int = 3,
    ) -> str:
        payload = RecommendationRequestSchema(
            title=title,
            abstract=abstract,
            keywords=keywords or [],
            study_program=study_program,
            research_area=research_area,
            mentor=mentor,
            mentor_topk=mentor_topk,
            exclude_professors=exclude_professors or [],
            include_professors=include_professors or [],
            alternatives=alternatives,
        )
        try:
            response = await generate_committee_recommendation(payload, db)
        except ActiveStaffDirectoryUnavailableError:
            return _error_json(
                "Active staff directory is unavailable; cannot safely recommend a committee.",
            )
        except InactiveStaffRequestedError as exc:
            return _inactive_staff_error(exc)
        except NoSimilarDefensesError:
            return _error_json(
                "No similar defended theses found to base a recommendation on.",
            )
        return response.model_dump_json(indent=2)

    return [
        StructuredTool.from_function(
            coroutine=recommend_diploma_committee,
            name=_RECOMMENDATION_TOOL_NAME,
            description=_RECOMMENDATION_TOOL_DESCRIPTION,
            args_schema=RecommendationRequestSchema,
        ),
    ]
