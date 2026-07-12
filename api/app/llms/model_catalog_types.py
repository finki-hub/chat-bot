from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CatalogProvider = Literal["openai", "google", "anthropic", "ollama"]
CatalogSource = Literal["live", "stale", "snapshot"]


class ExecutionPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    reasoning: bool
    sampling: bool
    tool_call: bool
    structured_output: bool


class ModelCapabilities(BaseModel):
    model_config = ConfigDict(frozen=True)

    reasoning: bool | None = None
    tool_call: bool | None = None
    structured_output: bool | None = None
    temperature: bool | None = None


class ModelModalities(BaseModel):
    model_config = ConfigDict(frozen=True)

    input: tuple[str, ...] = ()
    output: tuple[str, ...] = ()


class ModelLimits(BaseModel):
    model_config = ConfigDict(frozen=True)

    context: int | None = Field(default=None, ge=1)
    input: int | None = Field(default=None, ge=1)
    output: int | None = Field(default=None, ge=1)


class ModelPricing(BaseModel):
    model_config = ConfigDict(frozen=True)

    input: float | None = Field(default=None, ge=0)
    output: float | None = Field(default=None, ge=0)


class DisplayMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str | None = None
    capabilities: ModelCapabilities | None = None
    modalities: ModelModalities | None = None
    limits: ModelLimits | None = None
    pricing: ModelPricing | None = None
    status: str | None = None


class ModelDescriptor(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    provider: CatalogProvider
    name: str
    description: str | None = None
    execution: ExecutionPolicy
    capabilities: ModelCapabilities | None = None
    modalities: ModelModalities | None = None
    limits: ModelLimits | None = None
    pricing: ModelPricing | None = None
    status: str | None = None
    loaded: bool | None = None


class OllamaCatalogModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    loaded: bool | None


class ModelCatalogResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: Literal[1] = 1
    source: CatalogSource
    models: tuple[ModelDescriptor, ...]


class SnapshotEntry(DisplayMetadata):
    id: str


class SnapshotCatalog(BaseModel):
    model_config = ConfigDict(frozen=True)

    models: tuple[SnapshotEntry, ...]
