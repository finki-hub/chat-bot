from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from app.data.chat_persistence import ChatPersistenceDatabase
from app.data.chat_sharing import (
    create_conversation_share,
    get_conversation_share_status,
    load_shared_conversation,
    revoke_conversation_share,
)
from app.data.db import get_db
from app.schemas.chat_persistence import (
    ChatConversationShare,
    ChatConversationWithMessages,
)
from app.schemas.chat_state import UserScopedRequest
from app.utils.auth import verify_api_key

db_dep = Depends(get_db)
UserIdQuery = Annotated[UUID, Query()]

router = APIRouter(
    prefix="/chat/state",
    tags=["Chat State"],
    dependencies=[db_dep, Depends(verify_api_key)],
)


def _not_found() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Conversation not found",
    )


@router.post(
    "/conversations/{conversation_id}/share",
    status_code=status.HTTP_200_OK,
    operation_id="createChatStateConversationShare",
)
async def create_conversation_share_state(
    conversation_id: UUID,
    payload: UserScopedRequest,
    db: ChatPersistenceDatabase = db_dep,
) -> ChatConversationShare:
    share_token = await create_conversation_share(
        db,
        conversation_id=conversation_id,
        user_id=payload.user_id,
    )
    if share_token is None:
        raise _not_found()
    return ChatConversationShare(share_token=share_token)


@router.get(
    "/conversations/{conversation_id}/share",
    status_code=status.HTTP_200_OK,
    response_model=None,
    operation_id="getChatStateConversationShareStatus",
)
async def get_conversation_share_status_state(
    conversation_id: UUID,
    user_id: UserIdQuery,
    db: ChatPersistenceDatabase = db_dep,
) -> Response:
    is_shared = await get_conversation_share_status(
        db,
        conversation_id=conversation_id,
        user_id=user_id,
    )
    if is_shared is None:
        raise _not_found()
    return Response(
        status_code=(status.HTTP_200_OK if is_shared else status.HTTP_204_NO_CONTENT),
    )


@router.delete(
    "/conversations/{conversation_id}/share",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="revokeChatStateConversationShare",
)
async def revoke_conversation_share_state(
    conversation_id: UUID,
    payload: UserScopedRequest,
    db: ChatPersistenceDatabase = db_dep,
) -> None:
    revoked = await revoke_conversation_share(
        db,
        conversation_id=conversation_id,
        user_id=payload.user_id,
    )
    if not revoked:
        raise _not_found()


@router.get(
    "/shared/{share_token}",
    status_code=status.HTTP_200_OK,
    operation_id="loadSharedChatStateConversation",
)
async def load_shared_conversation_state(
    share_token: UUID,
    db: ChatPersistenceDatabase = db_dep,
) -> ChatConversationWithMessages:
    loaded = await load_shared_conversation(db, share_token=share_token)
    if loaded is None:
        raise _not_found()
    return loaded
