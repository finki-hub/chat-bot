import asyncio
import logging

from app.data.connection import Database
from app.data.questions import get_closest_questions
from app.llms.embeddings import generate_embeddings
from app.llms.models import Model
from app.llms.prompts import HYDE_SYSTEM_PROMPT
from app.llms.query_transform import transform_query
from app.utils.exceptions import RetrievalError
from app.utils.http_client import get_http_client
from app.utils.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()

RERANKER_MIN_SCORE: float = 0


async def get_retrieved_context(
    db: Database,
    query: str,
    embedding_model: Model,
    *,
    initial_k: int = 30,
    top_k: int = 10,
) -> str:
    """
    Retrieves context via two-stage retrieval: vector search over initial_k candidates
    followed by cross-encoder reranking, falling back to vector order on failure.
    """

    logger.info(
        "Retrieving context for query: '%s' with embedding model: %s",
        query,
        embedding_model,
    )

    rewritten_query, hyde_passage = await asyncio.gather(
        transform_query(
            query,
            Model.GPT_4_1_NANO,
            temperature=0.0,
            top_p=1.0,
            max_tokens=128,
        ),
        transform_query(
            query,
            Model.GPT_4_1_NANO,
            system_prompt=HYDE_SYSTEM_PROMPT,
            temperature=0.7,
            top_p=0.9,
            max_tokens=200,
        ),
    )

    logger.info("Transformed query: '%s'", rewritten_query)
    logger.info("HyDE passage: '%s'", hyde_passage)

    try:
        query_to_embed = (
            f"passage: {hyde_passage}"
            if embedding_model == Model.MULTILINGUAL_E5_LARGE
            else hyde_passage
        )
        prompt_embedding = await generate_embeddings(query_to_embed, embedding_model)
        initial_candidates = await get_closest_questions(
            db,
            prompt_embedding,
            embedding_model,
            limit=initial_k,
        )

        logger.info("Initial candidates retrieved: %d", len(initial_candidates))

        if not initial_candidates:
            return ""

    except Exception as e:
        raise RetrievalError("Failed during initial vector search") from e

    # rerank_docs omits links to match the indexed text
    # context_docs includes them for the LLM
    rerank_docs: list[str] = []
    context_docs: list[str] = []

    for q in initial_candidates:
        title_content = f"Наслов: {q.name}\nСодржина: {q.content}"
        rerank_docs.append(title_content)

        sources = (
            "Извори: " + ", ".join(f"{k}: {v}" for k, v in q.links.items())
            if q.links
            else None
        )
        full_doc = "\n".join(
            part
            for part in [f"Наслов: {q.name}", f"Содржина: {q.content}", sources]
            if part is not None
        )
        context_docs.append(full_doc)

    try:
        logger.info("Sending %d candidates to re-ranker...", len(rerank_docs))

        rerank_payload = {
            "query": rewritten_query,
            "documents": rerank_docs,
        }

        client = get_http_client()
        response = await client.post(
            f"{settings.GPU_API_URL}/rerank/",
            json=rerank_payload,
        )
        response.raise_for_status()

        ranked = response.json()["reranked_documents"]

        rerank_to_context = dict(zip(rerank_docs, context_docs, strict=False))

        final_docs: list[str] = [
            rerank_to_context.get(doc, doc)
            for item in ranked
            if item["score"] >= RERANKER_MIN_SCORE
            for doc in (str(item["document"]),)
        ][:top_k]

        logger.info(
            "Selected %d documents after reranking and score filtering",
            len(final_docs),
        )
    except Exception:
        logger.exception(
            "Reranking call failed. Using vector search order as a fallback",
        )
        final_docs = context_docs

    return "\n\n---\n\n".join(final_docs[:top_k])
