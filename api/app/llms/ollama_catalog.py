import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import anyio
import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from app.llms.model_catalog_types import OllamaCatalogModel
from app.llms.provider_credentials import require_provider_credential
from app.schemas.chat_credentials import OLLAMA_DEFAULT_BASE_URL, ChatCredentialSecret
from app.utils.http_client import get_http_client

logger = logging.getLogger(__name__)

_CAPABILITY_CONCURRENCY: Final = 8
_MAX_DISCOVERY_MODELS: Final = 100


class _OllamaTagModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    capabilities: tuple[str, ...] = ()


class _OllamaTagsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    models: tuple[_OllamaTagModel, ...] = ()


class _OllamaPsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    models: tuple[_OllamaTagModel, ...] = ()


class _OllamaShowResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    capabilities: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _DiscoveryContext:
    base_url: str
    headers: Mapping[str, str]
    timeout: httpx.Timeout
    limiter: anyio.CapacityLimiter


async def _fetch_capabilities(
    context: _DiscoveryContext,
    model_name: str,
) -> tuple[str, ...]:
    async with context.limiter:
        try:
            response = await get_http_client().post(
                f"{context.base_url}/api/show",
                headers=context.headers,
                json={"model": model_name},
                timeout=context.timeout,
            )
            response.raise_for_status()
            return _OllamaShowResponse.model_validate(
                response.json(),
            ).capabilities
        except (httpx.HTTPError, ValidationError, ValueError) as error:
            logger.warning(
                "Ollama model capability discovery failed: model=%s error=%s",
                model_name,
                type(error).__name__,
            )
            return ()


async def _fetch_loaded_models(
    context: _DiscoveryContext,
) -> frozenset[str] | None:
    try:
        response = await get_http_client().get(
            f"{context.base_url}/api/ps",
            headers=context.headers,
            timeout=context.timeout,
        )
        response.raise_for_status()
        parsed = _OllamaPsResponse.model_validate(response.json())
    except (httpx.HTTPError, ValidationError, ValueError) as error:
        logger.warning(
            "Ollama loaded-status discovery failed: %s",
            type(error).__name__,
        )
        return None
    return frozenset(model.name for model in parsed.models)


async def fetch_ollama_catalog(
    credential: ChatCredentialSecret,
) -> tuple[OllamaCatalogModel, ...]:
    credential = require_provider_credential("ollama", credential)
    context = _DiscoveryContext(
        base_url=(credential.base_url or OLLAMA_DEFAULT_BASE_URL).rstrip("/"),
        headers={"Authorization": f"Bearer {credential.api_key}"},
        timeout=httpx.Timeout(connect=3.0, read=5.0, write=5.0, pool=3.0),
        limiter=anyio.CapacityLimiter(_CAPABILITY_CONCURRENCY),
    )

    try:
        response = await get_http_client().get(
            f"{context.base_url}/api/tags",
            headers=context.headers,
            timeout=context.timeout,
        )
        response.raise_for_status()
        tags = _OllamaTagsResponse.model_validate(response.json())
    except (httpx.HTTPError, ValidationError, ValueError) as error:
        logger.warning("Ollama model discovery failed: %s", type(error).__name__)
        return ()

    models = tags.models[:_MAX_DISCOVERY_MODELS]
    if len(tags.models) > _MAX_DISCOVERY_MODELS:
        logger.warning(
            "Ollama model discovery truncated: count=%d limit=%d",
            len(tags.models),
            _MAX_DISCOVERY_MODELS,
        )

    loaded_models: frozenset[str] | None = None
    discovered_capabilities: dict[str, tuple[str, ...]] = {}
    async with anyio.create_task_group() as task_group:
        loaded_handle = task_group.create_task(_fetch_loaded_models(context))
        capability_handles = {
            model.name: task_group.create_task(
                _fetch_capabilities(context, model.name),
            )
            for model in models
            if not model.capabilities
        }
        loaded_models = await loaded_handle
        discovered_capabilities = {
            name: await handle for name, handle in capability_handles.items()
        }

    return tuple(
        OllamaCatalogModel(
            id=model.name,
            name=model.name,
            loaded=(None if loaded_models is None else model.name in loaded_models),
        )
        for model in models
        if "completion"
        in (model.capabilities or discovered_capabilities.get(model.name, ()))
    )
