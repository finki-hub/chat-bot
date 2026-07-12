import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Final

import anyio
import httpx
from pydantic import BaseModel, ValidationError

from app.llms.model_catalog_policy import MODEL_CATALOG, CatalogPolicy
from app.llms.model_catalog_remote import CatalogMetadataError, parse_models_dev
from app.llms.model_catalog_types import (
    CatalogSource,
    DisplayMetadata,
    ExecutionPolicy,
    ModelCatalogResponse,
    ModelDescriptor,
    OllamaCatalogModel,
    SnapshotCatalog,
)
from app.llms.models import Model
from app.utils.http_client import get_http_client

logger = logging.getLogger(__name__)

MODELS_DEV_URL: Final = "https://models.dev/api.json"
CATALOG_TTL_SECONDS: Final = 6 * 60 * 60
_SNAPSHOT_PATH: Final = Path(__file__).with_name("models_snapshot.json")

FetchMetadata = Callable[[], Awaitable[bytes]]
Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class CatalogFetchError(Exception):
    reason: str

    def __str__(self) -> str:
        return self.reason


async def fetch_models_dev() -> bytes:
    """Fetch models.dev once with bounded timeouts and no retries."""
    timeout = httpx.Timeout(connect=3.0, read=5.0, write=5.0, pool=3.0)
    try:
        response = await get_http_client().get(MODELS_DEV_URL, timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise CatalogFetchError(reason=type(error).__name__) from error
    return response.content


def _load_snapshot() -> dict[Model, DisplayMetadata]:
    try:
        snapshot = SnapshotCatalog.model_validate_json(_SNAPSHOT_PATH.read_bytes())
    except (OSError, ValidationError) as error:
        raise CatalogMetadataError(
            reason="bundled model snapshot is invalid",
        ) from error
    entries = {entry.id: entry for entry in snapshot.models}
    expected = tuple(policy.model.value for policy in MODEL_CATALOG)
    if not all(model_id in entries for model_id in expected):
        raise CatalogMetadataError(reason="bundled model snapshot is incomplete")
    return {
        policy.model: DisplayMetadata.model_validate(
            entries[policy.model.value].model_dump(exclude={"id"}),
        )
        for policy in MODEL_CATALOG
    }


def _descriptor(
    policy: CatalogPolicy,
    metadata: DisplayMetadata,
) -> ModelDescriptor:
    return ModelDescriptor(
        id=policy.model.value,
        provider=policy.provider,
        name=metadata.name,
        description=metadata.description,
        execution=policy.execution,
        capabilities=metadata.capabilities,
        modalities=metadata.modalities,
        limits=metadata.limits,
        pricing=metadata.pricing,
        status=metadata.status,
    )


def _ollama_descriptor(model: OllamaCatalogModel) -> ModelDescriptor:
    return ModelDescriptor(
        id=model.id,
        provider="ollama",
        name=model.name,
        execution=ExecutionPolicy(
            reasoning=False,
            sampling=True,
            tool_call=False,
            structured_output=False,
        ),
        loaded=model.loaded,
    )


def _merge_optional_metadata[MetadataT: BaseModel](
    snapshot: MetadataT | None,
    remote: MetadataT | None,
) -> MetadataT | None:
    if remote is None:
        return snapshot
    if snapshot is None:
        return remote
    return snapshot.model_copy(update=remote.model_dump(exclude_none=True))


def _merge_metadata(
    snapshot: DisplayMetadata,
    remote: DisplayMetadata | None,
) -> DisplayMetadata:
    if remote is None:
        return snapshot
    return DisplayMetadata(
        name=remote.name,
        description=(
            snapshot.description if remote.description is None else remote.description
        ),
        capabilities=_merge_optional_metadata(
            snapshot.capabilities,
            remote.capabilities,
        ),
        modalities=_merge_optional_metadata(snapshot.modalities, remote.modalities),
        limits=_merge_optional_metadata(snapshot.limits, remote.limits),
        pricing=_merge_optional_metadata(snapshot.pricing, remote.pricing),
        status=snapshot.status if remote.status is None else remote.status,
    )


class ModelCatalogService:
    """Mutable in-process TTL cache for immutable catalog responses."""

    def __init__(
        self,
        fetch_metadata: FetchMetadata = fetch_models_dev,
        clock: Clock = monotonic,
    ) -> None:
        self._fetch_metadata = fetch_metadata
        self._clock = clock
        self._snapshot = _load_snapshot()
        self._cached: ModelCatalogResponse | None = None
        self._expires_at = 0.0
        self._lock = anyio.Lock()

    def _build(
        self,
        source: CatalogSource,
        remote: dict[Model, DisplayMetadata],
        ollama_models: tuple[OllamaCatalogModel, ...] = (),
    ) -> ModelCatalogResponse:
        metadata = {
            model: _merge_metadata(snapshot, remote.get(model))
            for model, snapshot in self._snapshot.items()
        }
        return ModelCatalogResponse(
            source=source,
            models=(
                *(
                    _descriptor(policy, metadata[policy.model])
                    for policy in MODEL_CATALOG
                ),
                *(_ollama_descriptor(model) for model in ollama_models),
            ),
        )

    async def get_catalog(
        self,
        ollama_models: tuple[OllamaCatalogModel, ...] = (),
    ) -> ModelCatalogResponse:
        """Return live, cached, stale, or bundled catalog data without ever emptying it."""
        now = self._clock()
        if self._cached is not None and now < self._expires_at:
            return self._cached.model_copy(
                update={
                    "models": (
                        *self._cached.models,
                        *map(_ollama_descriptor, ollama_models),
                    ),
                },
            )

        async with self._lock:
            now = self._clock()
            if self._cached is not None and now < self._expires_at:
                return self._cached.model_copy(
                    update={
                        "models": (
                            *self._cached.models,
                            *map(_ollama_descriptor, ollama_models),
                        ),
                    },
                )
            try:
                payload = await self._fetch_metadata()
                remote = parse_models_dev(payload)
            except (CatalogFetchError, CatalogMetadataError) as error:
                if self._cached is not None:
                    logger.warning(
                        "model catalog refresh failed; source=stale error=%s",
                        error,
                    )
                    stale = self._cached.model_copy(update={"source": "stale"})
                    return stale.model_copy(
                        update={
                            "models": (
                                *stale.models,
                                *map(_ollama_descriptor, ollama_models),
                            ),
                        },
                    )
                logger.warning(
                    "model catalog refresh failed; source=snapshot error=%s",
                    error,
                )
                self._cached = self._build("snapshot", {})
                self._expires_at = now + CATALOG_TTL_SECONDS
                return self._cached.model_copy(
                    update={
                        "models": (
                            *self._cached.models,
                            *map(_ollama_descriptor, ollama_models),
                        ),
                    },
                )

            self._cached = self._build("live", remote)
            self._expires_at = now + CATALOG_TTL_SECONDS
            return self._cached.model_copy(
                update={
                    "models": (
                        *self._cached.models,
                        *map(_ollama_descriptor, ollama_models),
                    ),
                },
            )


model_catalog_service = ModelCatalogService()
