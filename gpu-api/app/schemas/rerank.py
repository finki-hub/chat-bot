from pydantic import BaseModel, Field


class RerankRequestSchema(BaseModel):
    query: str = Field(description="The original user query.")
    documents: list[str] = Field(
        description="A list of document contents to be reranked.",
    )


class RankedDocument(BaseModel):
    document: str = Field(description="The document text.")
    score: float = Field(
        description="Relevance score from the cross-encoder (higher is better).",
    )


class RerankResponseSchema(BaseModel):
    reranked_documents: list[RankedDocument] = Field(
        description="The documents reordered by relevance score, highest first.",
    )
