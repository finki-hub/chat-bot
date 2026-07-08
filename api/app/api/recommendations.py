import logging
from typing import assert_never

from fastapi import APIRouter, Depends, HTTPException, status

from app.data.connection import Database
from app.data.db import get_db
from app.schemas.recommendations import (
    RecommendationRequestSchema,
    RecommendationResponseSchema,
)
from app.services.recommendations import (
    ActiveStaffDirectoryUnavailableError,
    InactiveStaffRequestedError,
    NoSimilarDefensesError,
    generate_committee_recommendation,
)

logger = logging.getLogger(__name__)

db_dep = Depends(get_db)

router = APIRouter(
    prefix="/recommendations",
    tags=["Recommendations"],
    dependencies=[db_dep],
)


def _inactive_staff_detail(exc: InactiveStaffRequestedError) -> str:
    match exc.field:
        case "mentor":
            return "mentor must be an active staff member"
        case "include_professors":
            return (
                "include_professors must contain only active staff members: "
                f"{', '.join(exc.names)}"
            )
        case unreachable:
            assert_never(unreachable)
            msg = "Unexpected inactive staff field"
            raise AssertionError(msg)


@router.post(
    "/",
    summary="Recommend a thesis committee",
    description=(
        "Given a proposed thesis title plus optional abstract, keywords, study program, "
        "research area, constraints, and known mentor, recommend committee alternatives "
        "grounded in similar historical defenses and professor-paper expertise."
    ),
    status_code=status.HTTP_200_OK,
    operation_id="recommendCommittee",
)
async def recommend_committee(
    payload: RecommendationRequestSchema,
    db: Database = db_dep,
) -> RecommendationResponseSchema:
    try:
        return await generate_committee_recommendation(payload, db)
    except ActiveStaffDirectoryUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Active staff directory is unavailable; cannot safely recommend a committee.",
        ) from exc
    except InactiveStaffRequestedError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=_inactive_staff_detail(exc),
        ) from exc
    except NoSimilarDefensesError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No similar defended theses found to base a recommendation on.",
        ) from exc
