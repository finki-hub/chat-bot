import logging
from typing import TypedDict

import httpx

from app.utils.http_client import get_http_client
from app.utils.settings import Settings
from app.utils.timing import current_distinct_id, current_response_id

logger = logging.getLogger(__name__)
settings = Settings()

_RERANKER_TIMEOUT = httpx.Timeout(timeout=30.0)
_RERANKER_MAX_RETRIES: int = 1


class RerankPayload(TypedDict):
    query: str
    documents: list[str]


async def post_rerank(payload: RerankPayload) -> httpx.Response:
    client = get_http_client()
    for attempt in range(_RERANKER_MAX_RETRIES + 1):
        try:
            headers: dict[str, str] = {}
            rid = current_response_id()
            if rid:
                headers["X-Response-Id"] = rid
            did = current_distinct_id()
            if did:
                headers["X-Distinct-Id"] = did
            response = await client.post(
                f"{settings.GPU_API_URL}/rerank/",
                json=payload,
                headers=headers or None,
                timeout=_RERANKER_TIMEOUT,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            if attempt < _RERANKER_MAX_RETRIES:
                logger.warning(
                    "Reranker attempt %d failed (%s), retrying...",
                    attempt + 1,
                    exc,
                )
                continue
            raise
        else:
            return response
    raise RuntimeError(
        "Unreachable: reranker retry loop exited without return or raise",
    )
