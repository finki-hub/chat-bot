from datetime import UTC, datetime

from fastapi import APIRouter, Depends, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from app.data.connection import Database
from app.data.db import get_db
from app.schemas.health import DependencyStatus, HealthResponse, RootStatus

db_dep = Depends(get_db)

router = APIRouter(
    prefix="/health",
    tags=["Health"],
    dependencies=[db_dep],
)


@router.get(
    "/",
    summary="API Status",
    description="Returns a simple message if the API is up.",
    status_code=status.HTTP_200_OK,
    response_description="A one-line liveness confirmation",
    operation_id="getApiStatus",
)
@router.head("/", include_in_schema=False)
async def root() -> RootStatus:
    return RootStatus(message="The API is up and running.")


@router.get(
    "/health",
    summary="Application Health Check",
    description=(
        "Checks connectivity to the database and returns overall "
        "service health, timestamp, and each dependency's status."
    ),
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    response_description="Detailed health status including dependencies",
    operation_id="getHealthStatus",
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": HealthResponse,
            "description": "Service is unhealthy (one or more deps failed)",
            "content": {
                "application/json": {
                    "example": {
                        "status": "unhealthy",
                        "timestamp": "2025-06-05T12:34:56Z",
                        "dependencies": {
                            "database": {
                                "status": "database_query_error",
                                "healthy": False,
                            },
                        },
                    },
                },
            },
        },
    },
)
@router.head("/health", include_in_schema=False)
async def health_check(db: Database = db_dep) -> JSONResponse:
    db_status = "ok"
    healthy = True

    if not db.pool:
        db_status = "database_pool_not_initialized"
        healthy = False
    else:
        try:
            result = await db.fetchval("SELECT 1")
            if result != 1:
                db_status = "database_query_unexpected_result"
                healthy = False
        except Exception:
            db_status = "database_query_error"
            healthy = False

    overall = "ok" if healthy else "unhealthy"
    code = status.HTTP_200_OK if healthy else status.HTTP_503_SERVICE_UNAVAILABLE

    payload = HealthResponse(
        status=overall,
        timestamp=datetime.now(UTC),
        dependencies={"database": DependencyStatus(status=db_status, healthy=healthy)},
    )

    return JSONResponse(
        status_code=code,
        content=jsonable_encoder(payload.model_dump()),
    )
