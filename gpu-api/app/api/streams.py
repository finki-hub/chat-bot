import logging

from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse

from app.llms.streams import stream_response
from app.schemas.streams import StreamRequestSchema

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/stream",
    tags=["Stream"],
)


@router.post(
    "/",
    summary="Stream a chat response from a self-hosted model",
    description="Streams a chat response from a self-hosted model using the specified inference model and system prompt.",
    response_model=None,
    status_code=status.HTTP_200_OK,
    operation_id="selfHostedChat",
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "The requested model is not supported.",
        },
    },
)
async def stream(
    payload: StreamRequestSchema,
) -> StreamingResponse:
    logger.info(
        "Received stream request with prompt: %s, model: %s",
        payload.prompt,
        payload.inference_model,
    )

    system_prompt = (
        payload.system_prompt
        or "Ти си љубезен асистент кој помага на корисникот со неговите прашања."
    )

    return stream_response(
        user_prompt=payload.prompt,
        model=payload.inference_model,
        system_prompt=system_prompt,
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=payload.max_tokens,
    )
