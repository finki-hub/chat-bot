import json
import logging

from fastapi import APIRouter, Depends, status

from app.data.connection import Database
from app.data.db import get_db
from app.data.professor_groups import get_professor_groups
from app.schemas.diplomas import ProfessorGroupSchema

logger = logging.getLogger(__name__)

db_dep = Depends(get_db)

router = APIRouter(
    prefix="/groups",
    tags=["Groups"],
    dependencies=[db_dep],
)


@router.get(
    "/",
    summary="List temporal staff groups",
    description=(
        "Cohorts of professors who repeatedly worked together within a time window "
        "(community detection over the co-occurrence graph, per window). Filter by source "
        "('defense' = committee co-occurrence, 'coauthor' = paper co-authorship), by a year "
        "that falls inside the window, and/or by a professor (every window their cohort "
        "appears in). Precomputed by scripts/compute_professor_groups.py."
    ),
    status_code=status.HTTP_200_OK,
    operation_id="listProfessorGroups",
)
async def list_groups(
    source: str | None = None,
    year: int | None = None,
    professor: str | None = None,
    db: Database = db_dep,
) -> list[ProfessorGroupSchema]:
    rows = await get_professor_groups(
        db,
        source=source,
        year=year,
        professor=professor,
    )

    groups: list[ProfessorGroupSchema] = []
    for row in rows:
        members = row["members"]
        if isinstance(members, str):
            members = json.loads(members)
        groups.append(
            ProfessorGroupSchema(
                source=row["source"],
                window_start=row["window_start"],
                window_end=row["window_end"],
                group_index=row["group_index"],
                members=members,
                size=row["size"],
                min_weight=row["min_weight"],
            ),
        )
    return groups
