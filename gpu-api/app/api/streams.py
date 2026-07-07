import logging
from typing import Final

from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse

from app.llms.streams import stream_response
from app.schemas.streams import StreamRequestSchema

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT: Final[str] = (
    "Ти си ФИНКИ Хаб бот — стручен асистент за ФИНКИ "
    "(Факултет за информатички науки и компјутерско инженерство при УКИМ — "
    "Скопје). Одговарај исклучиво на македонски јазик и кирилично писмо, "
    "со јасни, точни и концизни одговори поврзани со ФИНКИ, студирањето "
    "на ФИНКИ или ФИНКИ Хаб. Буквално зачувај само URL-адреси, команди, "
    "кодови, кратенки, имиња на системи, предмети, професори/асистенти "
    "и други официјални идентификатори кога се потребни. Не измислувај "
    "факти, бројки, рокови или прописи."
)

router = APIRouter(
    prefix="/stream",
    tags=["Stream"],
)


@router.post(
    "/",
    summary="Stream a chat response from a self-hosted model",
    description="Streams a chat response from a self-hosted model using the configured system prompt.",
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
        "Received stream request: model=%s prompt_len=%d",
        payload.inference_model.value,
        len(payload.prompt),
    )

    return stream_response(
        user_prompt=payload.prompt,
        model=payload.inference_model,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=payload.max_tokens,
    )
