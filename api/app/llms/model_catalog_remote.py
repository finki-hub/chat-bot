from dataclasses import dataclass

from pydantic import (
    BaseModel,
    ConfigDict,
    JsonValue,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
)
from pydantic.type_adapter import TypeAdapter

from app.llms.model_catalog_policy import MODEL_CATALOG
from app.llms.model_catalog_types import (
    DisplayMetadata,
    ModelCapabilities,
    ModelLimits,
    ModelModalities,
    ModelPricing,
)
from app.llms.models import Model


@dataclass(frozen=True, slots=True)
class CatalogMetadataError(Exception):
    reason: str

    def __str__(self) -> str:
        return self.reason


class _RemoteModalities(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    input: tuple[StrictStr, ...] = ()
    output: tuple[StrictStr, ...] = ()


class _RemoteLimits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    context: StrictInt | None = None
    input: StrictInt | None = None
    output: StrictInt | None = None


class _RemoteCost(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    input: StrictFloat | None = None
    output: StrictFloat | None = None


class _RemoteModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: StrictStr
    name: StrictStr
    description: StrictStr | None = None
    reasoning: StrictBool | None = None
    tool_call: StrictBool | None = None
    structured_output: StrictBool | None = None
    temperature: StrictBool | None = None
    modalities: _RemoteModalities | None = None
    limit: _RemoteLimits | None = None
    cost: _RemoteCost | None = None
    status: StrictStr | None = None


class _RemoteProvider(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: StrictStr
    name: StrictStr
    models: dict[str, JsonValue]


_PAYLOAD_ADAPTER = TypeAdapter(dict[str, JsonValue])


def _metadata(remote: _RemoteModel) -> DisplayMetadata:
    capabilities = ModelCapabilities(
        reasoning=remote.reasoning,
        tool_call=remote.tool_call,
        structured_output=remote.structured_output,
        temperature=remote.temperature,
    )
    modalities = (
        None
        if remote.modalities is None
        else ModelModalities(
            input=remote.modalities.input,
            output=remote.modalities.output,
        )
    )
    limits = (
        None
        if remote.limit is None
        else ModelLimits(
            context=remote.limit.context,
            input=remote.limit.input,
            output=remote.limit.output,
        )
    )
    pricing = (
        None
        if remote.cost is None
        else ModelPricing(input=remote.cost.input, output=remote.cost.output)
    )
    return DisplayMetadata(
        name=remote.name,
        description=remote.description,
        capabilities=capabilities,
        modalities=modalities,
        limits=limits,
        pricing=pricing,
        status=remote.status,
    )


def parse_models_dev(payload: bytes) -> dict[Model, DisplayMetadata]:
    """Parse approved models from the provider-keyed models.dev response."""
    try:
        providers = _PAYLOAD_ADAPTER.validate_json(payload)
    except ValidationError as error:
        raise CatalogMetadataError(
            reason="models.dev payload is not valid JSON",
        ) from error

    result: dict[Model, DisplayMetadata] = {}
    for policy in MODEL_CATALOG:
        raw_provider = providers.get(policy.provider)
        if raw_provider is None:
            continue
        try:
            provider = _RemoteProvider.model_validate(raw_provider)
        except ValidationError:
            continue
        if provider.id != policy.provider:
            continue
        raw_model = provider.models.get(policy.model.value)
        if raw_model is None:
            continue
        try:
            remote = _RemoteModel.model_validate(raw_model)
        except ValidationError:
            continue
        if remote.id != policy.model.value:
            continue
        try:
            result[policy.model] = _metadata(remote)
        except ValidationError:
            continue
    return result
