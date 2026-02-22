import logging

from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse

from app.data.connection import Database
from app.data.db import get_db
from app.llms.chat import handle_chat
from app.llms.context import get_retrieved_context
from app.llms.models import Model
from app.schemas.chat import ChatSchema

logger = logging.getLogger(__name__)

db_dep = Depends(get_db)

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
    dependencies=[db_dep],
)


@router.post(
    "/",
    summary="Stream a chat response",
    description=(
        "Compute an embedding for the incoming question, retrieve top-N "
        "similar questions for context, construct a prompt, and stream back "
        "the LLM's answer as a text stream."
    ),
    response_class=StreamingResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Chunked stream of SSE events",
            "content": {
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "example": "data: Hello\n\ndata: World\n\n",
                    },
                },
            },
        },
    },
    operation_id="chatWithModel",
)
async def chat(
    payload: ChatSchema,
    db: Database = db_dep,
) -> StreamingResponse:
    logger.info(
        "Received chat request with payload: %s",
        payload.model_dump(mode="json", exclude_defaults=True),
    )

    context = await get_retrieved_context(
        db=db,
        query=payload.prompt,
        embedding_model=payload.embeddings_model,
    )

    if not context:
        context = "Не можев да пронајдам релевантни информации во базата на податоци."

    return await handle_chat(payload, context)


@router.get(
    "/models",
    summary="List available LLM models",
    description="Retrieve a list of all available LLM models for chat.",
    response_model=list[str],
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "List of available LLM models",
            "content": {
                "application/json": {
                    "example": ["llama3.3:70b", "qwen2.5:72b"],
                },
            },
        },
    },
)
def list_models() -> list[str]:
    return [m.value for m in Model]
