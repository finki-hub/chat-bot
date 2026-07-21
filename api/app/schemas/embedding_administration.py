from pydantic import BaseModel, ConfigDict


class EmbeddingCorpusCountResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    ready: int
    dirty: int


class EmbeddingLifecycleCountsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: EmbeddingCorpusCountResponse
    chunk: EmbeddingCorpusCountResponse
    diploma: EmbeddingCorpusCountResponse
    professor_document: EmbeddingCorpusCountResponse


class EmbeddingHealthResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    counts: EmbeddingLifecycleCountsResponse


class EmbeddingFillDirtyResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    counts: EmbeddingLifecycleCountsResponse


class EmbeddingRebuildResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    counts: EmbeddingLifecycleCountsResponse
