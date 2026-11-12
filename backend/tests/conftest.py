"""Shared fixtures.

The default test stack is fully in-process and deterministic:
SQLite (or ``TEST_DATABASE_URL`` for PostgreSQL in CI) + hashing embedder +
in-memory vector store + lexical reranker + fake LLM, with synchronous ingestion.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.container import Container, build_container
from app.core.config import (
    EmbeddingProvider,
    LLMProvider,
    RerankerProvider,
    Settings,
    VectorStoreProvider,
)
from app.main import create_app
from app.rag.llm import FakeLLM


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    db_url = os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    values: dict[str, object] = {
        "environment": "test",
        "database_url": db_url,
        "upload_dir": tmp_path / "uploads",
        "vector_store": VectorStoreProvider.MEMORY,
        "embedding_provider": EmbeddingProvider.HASHING,
        "reranker_provider": RerankerProvider.LEXICAL,
        "llm_provider": LLMProvider.FAKE,
        "chunk_size_tokens": 120,
        "chunk_overlap_tokens": 20,
        "log_level": "WARNING",
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def container(settings: Settings, fake_llm: FakeLLM) -> Iterator[Container]:
    c = build_container(settings, llm=fake_llm, synchronous_ingestion=True)
    if os.environ.get("TEST_DATABASE_URL"):
        c.db.drop_schema()  # clean slate on a shared PostgreSQL database
    yield c
    if os.environ.get("TEST_DATABASE_URL"):
        c.db.drop_schema()


@pytest.fixture
def client(settings: Settings, container: Container) -> Iterator[TestClient]:
    app = create_app(settings, container=container)
    with TestClient(app) as test_client:
        yield test_client
