from pydantic import BaseModel, Field


class RerankRequestSchema(BaseModel):
    query: str = Field(description="The original user query.")
    documents: list[str] = Field(
        description="A list of document contents to be reranked.",
    )


class RankedDocument(BaseModel):
    index: int = Field(
        description="Position of this document in the original request list, so the "
        "caller can map it back without matching on the (re-serialized) text.",
    )
    document: str = Field(description="The document text.")
    score: float = Field(
        description="Relevance score from the cross-encoder (higher is better).",
    )


class RerankResponseSchema(BaseModel):
    reranked_documents: list[RankedDocument] = Field(
        description="The documents reordered by relevance score, highest first.",
    )
