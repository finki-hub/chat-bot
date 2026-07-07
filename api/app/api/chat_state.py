from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.data.chat_persistence import (
    ChatMessageConflictError,
    ChatPersistenceDatabase,
    clear_active_stream_if_current,
    clear_stale_active_streams,
    create_conversation,
    get_conversation_owner,
    load_conversation,
    set_active_stream,
    upsert_message,
)
from app.data.chat_state import (
    mark_active_stream_stopped_if_current,
    upsert_assistant_message_by_response_id,
)
from app.data.chat_users import upsert_google_chat_user
from app.data.db import get_db
from app.schemas.chat_persistence import (
    ChatConversation,
    ChatConversationCreate,
    ChatConversationWithMessages,
    ChatMessage,
    ChatMessageRole,
    ChatMessageUpsert,
)
from app.schemas.chat_state import (
    AssistantMessageUpsertRequest,
    ClearStaleActiveStreamsRequest,
    ClearStaleActiveStreamsResponse,
    SetActiveStreamRequest,
    UserMessageUpsertRequest,
    UserScopedRequest,
)
from app.schemas.chat_user import ChatUser, ChatUserUpsert
from app.utils.auth import verify_api_key

db_dep = Depends(get_db)
api_key_dep = Depends(verify_api_key)
UserIdQuery = Annotated[UUID, Query()]

router = APIRouter(
    prefix="/chat/state",
    tags=["Chat State"],
    dependencies=[db_dep, api_key_dep],
)


def _not_found() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Conversation not found",
    )


async def _ensure_owned(
    db: ChatPersistenceDatabase,
    *,
    conversation_id: UUID,
    user_id: UUID,
) -> None:
    owner = await get_conversation_owner(db, conversation_id)
    if owner != user_id:
        raise _not_found()


@router.post(
    "/users/google",
    status_code=status.HTTP_200_OK,
    operation_id="upsertGoogleChatUser",
)
async def upsert_google_user_state(
    payload: ChatUserUpsert,
    db: ChatPersistenceDatabase = db_dep,
) -> ChatUser:
    return await upsert_google_chat_user(db, payload)


@router.post(
    "/conversations",
    status_code=status.HTTP_200_OK,
    operation_id="upsertChatStateConversation",
)
async def upsert_conversation_state(
    payload: ChatConversationCreate,
    db: ChatPersistenceDatabase = db_dep,
) -> ChatConversation:
    owner = await get_conversation_owner(db, payload.id)
    if owner is not None and owner != payload.user_id:
        raise _not_found()
    return await create_conversation(db, payload)


@router.get(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_200_OK,
    operation_id="loadChatStateConversation",
)
async def load_conversation_state(
    conversation_id: UUID,
    user_id: UserIdQuery,
    db: ChatPersistenceDatabase = db_dep,
) -> ChatConversationWithMessages:
    loaded = await load_conversation(
        db,
        conversation_id=conversation_id,
        user_id=user_id,
    )
    if loaded is None:
        raise _not_found()
    return loaded


@router.post(
    "/conversations/{conversation_id}/messages/user",
    status_code=status.HTTP_200_OK,
    operation_id="upsertChatStateUserMessage",
)
async def upsert_user_message_state(
    conversation_id: UUID,
    payload: UserMessageUpsertRequest,
    db: ChatPersistenceDatabase = db_dep,
) -> ChatMessage:
    await _ensure_owned(db, conversation_id=conversation_id, user_id=payload.user_id)
    try:
        return await upsert_message(
            db,
            ChatMessageUpsert(
                id=payload.id,
                conversation_id=conversation_id,
                role=ChatMessageRole.USER,
                content=payload.content,
                metadata=payload.metadata,
            ),
        )
    except ChatMessageConflictError as error:
        raise _not_found() from error


@router.put(
    "/conversations/{conversation_id}/messages/assistant/{response_id}",
    status_code=status.HTTP_200_OK,
    operation_id="upsertChatStateAssistantMessage",
)
async def upsert_assistant_message_state(
    conversation_id: UUID,
    response_id: UUID,
    payload: AssistantMessageUpsertRequest,
    db: ChatPersistenceDatabase = db_dep,
) -> ChatMessage:
    await _ensure_owned(db, conversation_id=conversation_id, user_id=payload.user_id)
    return await upsert_assistant_message_by_response_id(
        db,
        ChatMessageUpsert(
            id=payload.id,
            conversation_id=conversation_id,
            role=ChatMessageRole.ASSISTANT,
            content=payload.content,
            response_id=response_id,
            metadata=payload.metadata,
        ),
    )


@router.put(
    "/conversations/{conversation_id}/active-stream",
    status_code=status.HTTP_200_OK,
    operation_id="setChatStateActiveStream",
)
async def set_active_stream_state(
    conversation_id: UUID,
    payload: SetActiveStreamRequest,
    db: ChatPersistenceDatabase = db_dep,
) -> ChatConversation:
    updated = await set_active_stream(
        db,
        conversation_id=conversation_id,
        user_id=payload.user_id,
        active_stream_id=payload.active_stream_id,
        active_response_id=payload.active_response_id,
        active_status=payload.active_status,
    )
    if updated is None:
        raise _not_found()
    return updated


@router.delete(
    "/conversations/{conversation_id}/active-stream/{active_stream_id}",
    status_code=status.HTTP_200_OK,
    operation_id="clearChatStateActiveStreamIfCurrent",
)
async def clear_active_stream_state(
    conversation_id: UUID,
    active_stream_id: UUID,
    user_id: UserIdQuery,
    db: ChatPersistenceDatabase = db_dep,
) -> ChatConversation:
    updated = await clear_active_stream_if_current(
        db,
        conversation_id=conversation_id,
        user_id=user_id,
        active_stream_id=active_stream_id,
    )
    if updated is None:
        raise _not_found()
    return updated


@router.post(
    "/conversations/{conversation_id}/active-stream/{active_stream_id}/stop",
    status_code=status.HTTP_200_OK,
    operation_id="stopChatStateActiveStreamIfCurrent",
)
async def stop_active_stream_state(
    conversation_id: UUID,
    active_stream_id: UUID,
    payload: UserScopedRequest,
    db: ChatPersistenceDatabase = db_dep,
) -> ChatConversation:
    updated = await mark_active_stream_stopped_if_current(
        db,
        conversation_id=conversation_id,
        user_id=payload.user_id,
        active_stream_id=active_stream_id,
    )
    if updated is None:
        raise _not_found()
    return updated


@router.post(
    "/active-streams/clear-stale",
    status_code=status.HTTP_200_OK,
    operation_id="clearStaleChatStateActiveStreams",
)
async def clear_stale_active_stream_state(
    payload: ClearStaleActiveStreamsRequest,
    db: ChatPersistenceDatabase = db_dep,
) -> ClearStaleActiveStreamsResponse:
    cleared_count = await clear_stale_active_streams(
        db,
        stale_before=payload.stale_before,
    )
    return ClearStaleActiveStreamsResponse(cleared_count=cleared_count)
