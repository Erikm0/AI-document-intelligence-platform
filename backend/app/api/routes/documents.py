from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Query, Response, UploadFile, status
from fastapi.responses import FileResponse

from app.api.deps import ContainerDep, DocumentServiceDep, SessionDep
from app.core.exceptions import DocumentNotFoundError
from app.db.models import DocumentStatus
from app.repositories.documents import DocumentRepository
from app.schemas.common import ERROR_RESPONSES, ErrorResponse, Page
from app.schemas.documents import ChunkRead, DocumentRead, DocumentStats

router = APIRouter(prefix="/documents", tags=["documents"], responses=ERROR_RESPONSES)


@router.post(
    "",
    response_model=DocumentRead,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document",
    description=(
        "Upload a PDF, DOCX or TXT file. The file is validated (extension, magic bytes, "
        "size, duplicate hash) and stored, then parsed, chunked and embedded "
        "asynchronously. Poll `GET /documents/{id}` until `status` is `ready`."
    ),
    responses={
        409: {"model": ErrorResponse, "description": "Duplicate document"},
        413: {"model": ErrorResponse, "description": "File too large"},
        415: {"model": ErrorResponse, "description": "Unsupported file type"},
    },
)
def upload_document(
    file: Annotated[UploadFile, File(description="PDF, DOCX or TXT file")],
    service: DocumentServiceDep,
    session: SessionDep,
    container: ContainerDep,
) -> DocumentRead:
    document = service.create_from_upload(file.filename, file.file)
    session.commit()  # make the row visible to the ingestion worker
    container.queue.submit(document.id)
    session.refresh(document)
    return DocumentRead.model_validate(document)


@router.get("", response_model=Page[DocumentRead], summary="List documents")
def list_documents(
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    status_filter: Annotated[DocumentStatus | None, Query(alias="status")] = None,
    search: Annotated[str | None, Query(max_length=200)] = None,
) -> Page[DocumentRead]:
    rows, total = DocumentRepository(session).paginate(
        limit=limit, offset=offset, status=status_filter, search=search
    )
    return Page[DocumentRead](
        items=[DocumentRead.model_validate(r) for r in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/stats", response_model=DocumentStats, summary="Corpus statistics")
def document_stats(session: SessionDep) -> DocumentStats:
    return DocumentStats(**DocumentRepository(session).stats())


@router.get("/{document_id}", response_model=DocumentRead, summary="Get a document")
def get_document(document_id: uuid.UUID, service: DocumentServiceDep) -> DocumentRead:
    return DocumentRead.model_validate(service.get(document_id))


@router.get(
    "/{document_id}/chunks", response_model=Page[ChunkRead], summary="List a document's chunks"
)
def list_chunks(
    document_id: uuid.UUID,
    service: DocumentServiceDep,
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[ChunkRead]:
    service.get(document_id)
    rows, total = DocumentRepository(session).list_chunks(document_id, limit=limit, offset=offset)
    return Page[ChunkRead](
        items=[ChunkRead.model_validate(r) for r in rows], total=total, limit=limit, offset=offset
    )


@router.get(
    "/{document_id}/file",
    summary="Download the original file",
    response_class=FileResponse,
    responses={200: {"content": {"application/octet-stream": {}}}},
)
def download_document(document_id: uuid.UUID, service: DocumentServiceDep) -> FileResponse:
    document = service.get(document_id)
    path = Path(document.storage_path)
    if not path.exists():
        raise DocumentNotFoundError("The original file is no longer available on disk")
    return FileResponse(path, media_type=document.content_type, filename=document.filename)


@router.post(
    "/{document_id}/reprocess",
    response_model=DocumentRead,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Re-run ingestion (e.g. after changing chunking or embedding settings)",
)
def reprocess_document(
    document_id: uuid.UUID,
    service: DocumentServiceDep,
    session: SessionDep,
    container: ContainerDep,
) -> DocumentRead:
    document = service.mark_for_reprocessing(document_id)
    session.commit()
    container.queue.submit(document.id)
    session.refresh(document)
    return DocumentRead.model_validate(document)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete a document, its chunks, vectors and file",
)
def delete_document(document_id: uuid.UUID, service: DocumentServiceDep) -> Response:
    service.delete(document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
