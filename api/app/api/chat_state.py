from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.data.chat_conversation_delete import delete_conversation, delete_conversations
from app.data.chat_persistence import (
    ChatMessageConflictError,
    ChatPersistenceDatabase,
    clear_active_stream_if_current,
    clear_stale_active_streams,
    create_conversation,
    get_conversation_owner,
    list_conversations,
    load_conversation,
    set_active_stream,
    update_conversation,
    upsert_message,
)
from app.data.chat_state import (
    mark_active_stream_stopped_if_current,
    replace_assistant_message_and_prune_after,
    upsert_assistant_message_by_response_id,
)
from app.data.chat_users import upsert_chat_user
from app.data.db import get_db
from app.schemas.chat_persistence import (
    ChatConversation,
    ChatConversationCreate,
    ChatConversationUpdate,
    ChatConversationWithMessages,
    ChatMessage,
    ChatMessageRole,
    ChatMessageUpsert,
)
from app.schemas.chat_state import (
    AssistantMessageReplacementRequest,
    AssistantMessageUpsertRequest,
    ClearStaleActiveStreamsRequest,
    ClearStaleActiveStreamsResponse,
    ConversationUpdateRequest,
    SetActiveStreamRequest,
    UserMessageUpsertRequest,
    UserScopedRequest,
)
from app.schemas.chat_user import ChatUser, ChatUserUpsert
from app.utils.auth import verify_api_key

db_dep = Depends(get_db)
api_key_dep = Depends(verify_api_key)
UserIdQuery = Annotated[UUID, Query()]
ConversationLimitQuery = Annotated[int, Query(ge=1, le=100)]

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
    "/users",
    status_code=status.HTTP_200_OK,
    operation_id="upsertChatUser",
)
async def upsert_user_state(
    payload: ChatUserUpsert,
    db: ChatPersistenceDatabase = db_dep,
) -> ChatUser:
    return await upsert_chat_user(db, payload)


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
    "/conversations",
    status_code=status.HTTP_200_OK,
    operation_id="listChatStateConversations",
)
async def list_conversation_state(
    user_id: UserIdQuery,
    limit: ConversationLimitQuery = 50,
    db: ChatPersistenceDatabase = db_dep,
) -> list[ChatConversation]:
    return await list_conversations(db, user_id=user_id, limit=limit)


@router.delete(
    "/conversations",
    status_code=status.HTTP_200_OK,
    operation_id="deleteChatStateConversations",
)
async def delete_conversation_state_all(
    user_id: UserIdQuery,
    db: ChatPersistenceDatabase = db_dep,
) -> list[ChatConversation]:
    return await delete_conversations(db, user_id=user_id)


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


@router.patch(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_200_OK,
    operation_id="updateChatStateConversation",
)
async def update_conversation_state(
    conversation_id: UUID,
    payload: ConversationUpdateRequest,
    db: ChatPersistenceDatabase = db_dep,
) -> ChatConversation:
    updated = await update_conversation(
        db,
        conversation_id=conversation_id,
        user_id=payload.user_id,
        update=ChatConversationUpdate(model=payload.model, title=payload.title),
    )
    if updated is None:
        raise _not_found()
    return updated


@router.delete(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_200_OK,
    operation_id="deleteChatStateConversation",
)
async def delete_conversation_state(
    conversation_id: UUID,
    user_id: UserIdQuery,
    db: ChatPersistenceDatabase = db_dep,
) -> ChatConversation:
    deleted = await delete_conversation(
        db,
        conversation_id=conversation_id,
        user_id=user_id,
    )
    if deleted is None:
        raise _not_found()
    return deleted


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
    "/conversations/{conversation_id}/messages/assistant/{message_id}/replacement/{response_id}",
    status_code=status.HTTP_200_OK,
    operation_id="replaceChatStateAssistantMessage",
)
async def replace_assistant_message_state(
    conversation_id: UUID,
    message_id: UUID,
    response_id: UUID,
    payload: AssistantMessageReplacementRequest,
    db: ChatPersistenceDatabase = db_dep,
) -> ChatMessage:
    if message_id not in payload.retained_message_ids:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Replacement target must be retained",
        )
    updated = await replace_assistant_message_and_prune_after(
        db,
        ChatMessageUpsert(
            id=message_id,
            conversation_id=conversation_id,
            role=ChatMessageRole.ASSISTANT,
            content=payload.content,
            response_id=response_id,
            metadata=payload.metadata,
        ),
        retained_message_ids=payload.retained_message_ids,
        user_id=payload.user_id,
    )
    if updated is None:
        raise _not_found()
    return updated


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
