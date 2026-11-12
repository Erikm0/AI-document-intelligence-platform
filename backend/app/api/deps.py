from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.container import Container
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService


def get_container(request: Request) -> Container:
    return request.app.state.container  # type: ignore[no-any-return]


ContainerDep = Annotated[Container, Depends(get_container)]


def get_session(container: ContainerDep) -> Iterator[Session]:
    with container.db.session() as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]


def get_document_service(container: ContainerDep, session: SessionDep) -> DocumentService:
    return DocumentService(
        session,
        settings=container.settings,
        storage=container.storage,
        vector_store=container.vector_store,
        lexical_index=container.lexical_index,
    )


def get_chat_service(container: ContainerDep) -> ChatService:
    return ChatService(
        container.db, container.pipeline, history_turns=container.settings.history_turns
    )


DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
