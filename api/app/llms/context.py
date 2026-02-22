import logging

import httpx

from app.data.connection import Database
from app.data.questions import get_closest_questions
from app.llms.embeddings import generate_embeddings
from app.llms.models import Model
from app.llms.query_transform import transform_query
from app.utils.exceptions import RetrievalError
from app.utils.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()


async def get_retrieved_context(
    db: Database,
    query: str,
    embedding_model: Model,
    *,
    use_reranker: bool,
    initial_k: int = 30,
    top_k: int = 10,
) -> str:
    """
    Performs a retrieval process. If use_reranker is True, it's a two-stage
    process (vector search + re-ranking). Otherwise, it's a single-stage
    vector search.
    """

    logger.info(
        "Retrieving context for query: '%s' with embedding model: %s",
        query,
        embedding_model,
    )

    query = await transform_query(
        query,
        Model.GPT_4_1_NANO,
        temperature=0.0,
        top_p=1.0,
        max_tokens=128,
    )

    logger.info("Transformed query: '%s'", query)

    retrieval_limit = initial_k if use_reranker else top_k

    try:
        query_to_embed = (
            f"query: {query}"
            if embedding_model == Model.MULTILINGUAL_E5_LARGE
            else query
        )
        prompt_embedding = await generate_embeddings(query_to_embed, embedding_model)
        initial_candidates = await get_closest_questions(
            db,
            prompt_embedding,
            embedding_model,
            limit=retrieval_limit,
        )

        logger.info("Initial candidates retrieved: %d", len(initial_candidates))

        if not initial_candidates:
            return ""

    except Exception as e:
        raise RetrievalError("Failed during initial vector search") from e

    candidate_docs = [
        "\n".join(
            filter(
                None,
                [
                    f"Наслов: {q.name}",
                    f"Содржина: {q.content}",
                    ("Извори: " + ", ".join(f"{k}: {v}" for k, v in q.links.items()))
                    if q.links
                    else None,
                ],
            ),
        )
        for q in initial_candidates
    ]

    logger.info("Reranking enabled: %s", use_reranker)

    if use_reranker:
        try:
            logger.info("Sending %d candidates to re-ranker...", len(candidate_docs))

            rerank_payload = {
                "query": query,
                "documents": candidate_docs,
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{settings.GPU_API_URL}/rerank/",
                    json=rerank_payload,
                )

                response.raise_for_status()

                final_docs = response.json()["reranked_documents"]

            logger.info(
                "Selected top %d documents",
                len(final_docs),
            )
        except Exception:
            logger.exception(
                "Reranking call failed. Using vector search order as a fallback",
            )
            final_docs = candidate_docs
    else:
        final_docs = candidate_docs

    return "\n\n---\n\n".join(final_docs[:top_k])
