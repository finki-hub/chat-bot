import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.data.staff import StaffDirectoryUnavailableError, get_staff_directory
from app.schemas.professors import ProfessorSchema
from app.utils.auth import verify_api_key

logger = logging.getLogger(__name__)

api_key_dep = Depends(verify_api_key)

router = APIRouter(
    prefix="/professors",
    tags=["Professors"],
    dependencies=[api_key_dep],
)


@router.get(
    "/",
    summary="List active faculty staff",
    description=(
        "Return active staff from the Directus directory with the public profile "
        "fields the committee planner needs. Falls back to the legacy staff export "
        "when Directus is unavailable."
    ),
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_401_UNAUTHORIZED: {"description": "Invalid or missing API Key"},
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "Staff directory is unavailable",
        },
    },
    operation_id="listProfessors",
)
async def list_professors() -> list[ProfessorSchema]:
    try:
        staff = await get_staff_directory()
    except StaffDirectoryUnavailableError as exc:
        logger.exception("Staff directory is unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Staff directory is unavailable",
        ) from exc
    return [ProfessorSchema.model_validate(member) for member in staff]
