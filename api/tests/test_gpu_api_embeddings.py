import pytest
from pydantic import ValidationError

from app.llms import gpu_api
from app.llms.models import MODEL_EMBEDDING_DIMENSIONS, Model

MODEL = Model.BGE_M3_LOCAL
DIMENSIONS = MODEL_EMBEDDING_DIMENSIONS[MODEL]


class FakeResponse:
    def __init__(self, embeddings: object) -> None:
        self._embeddings = embeddings

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {"embeddings": self._embeddings}


class FakeClient:
    def __init__(self, embeddings: object) -> None:
        self._response = FakeResponse(embeddings)

    async def post(self, _url: str, **_kwargs: object) -> FakeResponse:
        return self._response


@pytest.mark.parametrize(
    ("text", "embeddings"),
    [
        ("single", [[0.1, 0.2]]),
        (["first", "second"], [0.1, 0.2]),
    ],
)
@pytest.mark.anyio
async def test_gpu_embeddings_reject_response_shape_mismatched_to_input(
    monkeypatch: pytest.MonkeyPatch,
    text: str | list[str],
    embeddings: object,
) -> None:
    # Given: the GPU service returns a valid embedding shape for the other input mode.
    monkeypatch.setattr(gpu_api, "get_http_client", lambda: FakeClient(embeddings))

    # When/Then: boundary parsing rejects the mismatched response shape.
    with pytest.raises(ValidationError):
        await gpu_api.generate_gpu_api_embeddings(text, MODEL)


@pytest.mark.parametrize(
    ("text", "embeddings"),
    [
        ("single", [0.0] * DIMENSIONS),
        (["first", "second"], [[0.0] * DIMENSIONS, [1.0] * DIMENSIONS]),
    ],
)
@pytest.mark.anyio
async def test_gpu_embeddings_accept_complete_response(
    monkeypatch: pytest.MonkeyPatch,
    text: str | list[str],
    embeddings: list[float] | list[list[float]],
) -> None:
    monkeypatch.setattr(gpu_api, "get_http_client", lambda: FakeClient(embeddings))

    result = await gpu_api.generate_gpu_api_embeddings(text, MODEL)

    assert result == embeddings


@pytest.mark.anyio
async def test_gpu_embeddings_reject_wrong_batch_cardinality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embeddings = [[0.0] * DIMENSIONS]
    monkeypatch.setattr(gpu_api, "get_http_client", lambda: FakeClient(embeddings))

    with pytest.raises(ValueError, match="1 embeddings for 2 inputs"):
        await gpu_api.generate_gpu_api_embeddings(["first", "second"], MODEL)


@pytest.mark.anyio
async def test_gpu_embeddings_reject_wrong_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_api, "get_http_client", lambda: FakeClient([0.0]))

    with pytest.raises(ValueError, match=f"1 dimensions; expected {DIMENSIONS}"):
        await gpu_api.generate_gpu_api_embeddings("single", MODEL)


@pytest.mark.parametrize("value", [True, "1.0", float("nan"), float("inf")])
@pytest.mark.anyio
async def test_gpu_embeddings_reject_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    value: object,
) -> None:
    embeddings: list[object] = [0.0] * DIMENSIONS
    embeddings[0] = value
    monkeypatch.setattr(gpu_api, "get_http_client", lambda: FakeClient(embeddings))

    with pytest.raises(ValidationError):
        await gpu_api.generate_gpu_api_embeddings("single", MODEL)
