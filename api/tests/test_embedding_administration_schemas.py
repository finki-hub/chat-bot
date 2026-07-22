from app.main import make_app
from app.utils.settings import Settings

API_KEY = "task-5-api-key"


def test_embedding_administration_openapi_has_typed_operation_schemas() -> None:
    # Given: the configured application.
    app = make_app(
        Settings(
            API_KEY=API_KEY,
            CREDENTIAL_ENCRYPTION_KEY="test-credential-key",
            MCP_API_KEY="test-mcp-key",
        ),
    )

    # When: FastAPI generates OpenAPI.
    schema = app.openapi()

    # Then: each stable route exposes its explicit typed response model.
    for path, method, response_name in (
        ("/embeddings/health", "get", "EmbeddingHealthResponse"),
        ("/embeddings/fill-dirty", "post", "EmbeddingFillDirtyResponse"),
        ("/embeddings/rebuild", "post", "EmbeddingRebuildResponse"),
    ):
        response = schema["paths"][path][method]["responses"]["200"]["content"][
            "application/json"
        ]["schema"]
        assert response == {"$ref": f"#/components/schemas/{response_name}"}
