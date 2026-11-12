from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Query, Response, status

from app.api.deps import SessionDep
from app.core.exceptions import ConversationNotFoundError
from app.db.models import Conversation
from app.repositories.conversations import ConversationRepository
from app.schemas.chat import (
    ConversationCreate,
    ConversationDetail,
    ConversationRead,
    ConversationUpdate,
    MessageRead,
)
from app.schemas.common import ERROR_RESPONSES, Page

router = APIRouter(prefix="/conversations", tags=["conversations"], responses=ERROR_RESPONSES)


def _get(repo: ConversationRepository, conversation_id: uuid.UUID, **kw: bool) -> Conversation:
    conversation = repo.get(conversation_id, **kw)
    if conversation is None:
        raise ConversationNotFoundError(f"Conversation {conversation_id} not found")
    return conversation


def _read(conversation: Conversation, count: int) -> ConversationRead:
    return ConversationRead(
        id=conversation.id,
        title=conversation.title,
        document_ids=[uuid.UUID(d) for d in conversation.document_ids or []],
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        message_count=count,
    )


@router.get("", response_model=Page[ConversationRead], summary="List conversations")
def list_conversations(
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[ConversationRead]:
    rows, total = ConversationRepository(session).paginate(limit=limit, offset=offset)
    return Page[ConversationRead](
        items=[_read(c, n) for c, n in rows], total=total, limit=limit, offset=offset
    )


@router.post(
    "",
    response_model=ConversationRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create an empty conversation",
)
def create_conversation(body: ConversationCreate, session: SessionDep) -> ConversationRead:
    conversation = ConversationRepository(session).create(
        body.title or "New conversation", body.document_ids
    )
    return _read(conversation, 0)


@router.get("/{conversation_id}", response_model=ConversationDetail, summary="Get a conversation")
def get_conversation(conversation_id: uuid.UUID, session: SessionDep) -> ConversationDetail:
    conversation = _get(ConversationRepository(session), conversation_id, with_messages=True)
    base = _read(conversation, len(conversation.messages))
    return ConversationDetail(
        **base.model_dump(),
        messages=[MessageRead.model_validate(m) for m in conversation.messages],
    )


@router.patch("/{conversation_id}", response_model=ConversationRead, summary="Rename / re-scope")
def update_conversation(
    conversation_id: uuid.UUID, body: ConversationUpdate, session: SessionDep
) -> ConversationRead:
    repo = ConversationRepository(session)
    conversation = _get(repo, conversation_id, with_messages=True)
    if body.title is not None:
        conversation.title = body.title
    if body.document_ids is not None:
        conversation.document_ids = [str(d) for d in body.document_ids]
    session.flush()
    return _read(conversation, len(conversation.messages))


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete a conversation",
)
def delete_conversation(conversation_id: uuid.UUID, session: SessionDep) -> Response:
    repo = ConversationRepository(session)
    repo.delete(_get(repo, conversation_id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)
