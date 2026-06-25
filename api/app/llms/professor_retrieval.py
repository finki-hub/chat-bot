import json
import logging

from app.data.connection import Database
from app.data.professor_documents import get_closest_professor_documents
from app.llms.embeddings import generate_embeddings
from app.llms.models import Model
from app.llms.text_utils import _prepare_text_for_embedding
from app.recommenders.recommend import RetrievedPaper

logger = logging.getLogger(__name__)

# Cross-lingual gap: Macedonian title vs English abstracts sits farther out than same-language matches; errs toward recall since expertise/buddy aggregate many papers.
PAPER_DISTANCE_CEILING = 0.8


async def retrieve_professor_papers(
    db: Database,
    query_text: str,
    model: Model,
    limit: int,
    *,
    threshold: float = PAPER_DISTANCE_CEILING,
    query_embedding: list[float] | None = None,
) -> list[RetrievedPaper]:
    embedded = (
        query_embedding
        if query_embedding is not None
        else await generate_embeddings(
            _prepare_text_for_embedding(query_text, model, is_document=False),
            model,
        )
    )

    rows = await get_closest_professor_documents(
        db,
        embedded,
        model,
        limit=limit,
        threshold=threshold,
    )

    papers: list[RetrievedPaper] = []
    for row in rows:
        authors = row["canonical_authors"]
        if isinstance(authors, str):
            authors = json.loads(authors)
        papers.append(
            RetrievedPaper(
                external_id=row["external_id"],
                title=row["title"],
                coauthors=tuple(authors or ()),
                distance=float(row["distance"]),
                year=row["year"],
            ),
        )
    return papers
