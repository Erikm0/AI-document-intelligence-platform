"""Data-access layer for conversations and messages."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.db.models import Conversation, Message, MessageRole, utcnow


class ConversationRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, title: str, document_ids: list[uuid.UUID] | None = None) -> Conversation:
        conversation = Conversation(
            title=title[:255], document_ids=[str(d) for d in (document_ids or [])]
        )
        self.session.add(conversation)
        self.session.flush()
        return conversation

    def get(
        self, conversation_id: uuid.UUID, *, with_messages: bool = False
    ) -> Conversation | None:
        if not with_messages:
            return self.session.get(Conversation, conversation_id)
        return self.session.scalar(
            select(Conversation)
            .where(Conversation.id == conversation_id)
            .options(selectinload(Conversation.messages))
        )

    def paginate(
        self, *, limit: int = 50, offset: int = 0
    ) -> tuple[Sequence[tuple[Conversation, int]], int]:
        total = self.session.scalar(select(func.count()).select_from(Conversation)) or 0
        msg_count = (
            select(Message.conversation_id, func.count(Message.id).label("n"))
            .group_by(Message.conversation_id)
            .subquery()
        )
        rows = self.session.execute(
            select(Conversation, func.coalesce(msg_count.c.n, 0))
            .outerjoin(msg_count, msg_count.c.conversation_id == Conversation.id)
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
            .offset(offset)
        ).all()
        return [(r[0], int(r[1])) for r in rows], total

    def delete(self, conversation: Conversation) -> None:
        self.session.delete(conversation)
        self.session.flush()

    def recent_messages(self, conversation_id: uuid.UUID, limit: int) -> list[Message]:
        if limit <= 0:
            return []
        rows = self.session.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.seq.desc())
            .limit(limit)
        ).all()
        return list(reversed(rows))

    def add_message(
        self,
        conversation: Conversation,
        role: MessageRole,
        content: str,
        *,
        citations: list[dict[str, Any]] | None = None,
        refused: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        last_seq = self.session.scalar(
            select(func.max(Message.seq)).where(Message.conversation_id == conversation.id)
        )
        message = Message(
            conversation_id=conversation.id,
            seq=(last_seq or 0) + 1,
            role=role,
            content=content,
            citations=citations or [],
            refused=refused,
            metadata_=metadata or {},
        )
        self.session.add(message)
        conversation.updated_at = utcnow()
        self.session.flush()
        return message
