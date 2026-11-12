from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class APIModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class ErrorBody(BaseModel):
    code: str = Field(examples=["document_not_found"])
    message: str = Field(examples=["Document 3f2a... was not found"])
    details: Any | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
    request_id: str | None = None


class Page(APIModel, Generic[T]):
    items: list[T]
    total: int
    limit: int
    offset: int


ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    404: {"model": ErrorResponse, "description": "Resource not found"},
    422: {"model": ErrorResponse, "description": "Validation error"},
    503: {"model": ErrorResponse, "description": "A dependency (LLM / vector store) is down"},
}
