from datetime import UTC, datetime

import torch
from fastapi import APIRouter, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from app.schemas.health import DependencyStatus, HealthResponse, RootStatus

router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


@router.get(
    "/",
    summary="Service Status",
    description="Simple liveness probe; returns 200 if the service is up.",
    status_code=status.HTTP_200_OK,
    response_description="One-line status message",
    operation_id="gpuApiStatus",
)
@router.head("/", include_in_schema=False)
async def root() -> RootStatus:
    return RootStatus(message="gpu-api is running.")


@router.get(
    "/health",
    summary="Detailed Health Check",
    description=(
        "Performs quick checks of external dependencies (CUDA device) "
        "and returns overall health plus each dependency's status."
    ),
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    response_description="Detailed health status",
    operation_id="gpuApiHealth",
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": HealthResponse,
            "description": "GPU device not available",
            "content": {
                "application/json": {
                    "example": {
                        "status": "unhealthy",
                        "timestamp": "2025-06-05T12:00:00Z",
                        "dependencies": {
                            "cuda": {"status": "cuda_not_available", "healthy": False},
                        },
                    },
                },
            },
        },
    },
)
@router.head("/health", include_in_schema=False)
async def health_check() -> JSONResponse:
    cuda_ok = torch.cuda.is_available()
    dep_status = DependencyStatus(
        status="ok" if cuda_ok else "cuda_not_available",
        healthy=cuda_ok,
    )

    overall = "ok" if cuda_ok else "unhealthy"
    code = status.HTTP_200_OK if cuda_ok else status.HTTP_503_SERVICE_UNAVAILABLE

    payload = HealthResponse(
        status=overall,
        timestamp=datetime.now(UTC),
        dependencies={"cuda": dep_status},
    )

    return JSONResponse(
        status_code=code,
        content=jsonable_encoder(payload.model_dump()),
    )
