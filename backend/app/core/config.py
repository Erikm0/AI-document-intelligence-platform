"""Application configuration.

All settings are read from environment variables (or a ``.env`` file) and validated
by Pydantic, so a misconfigured deployment fails fast at start-up instead of at the
first request.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingProvider(StrEnum):
    SENTENCE_TRANSFORMERS = "sentence-transformers"
    HASHING = "hashing"  # dependency-free fallback, used by tests and CI


class VectorStoreProvider(StrEnum):
    CHROMA_HTTP = "chroma-http"
    CHROMA_PERSISTENT = "chroma-persistent"
    MEMORY = "memory"


class RerankerProvider(StrEnum):
    CROSS_ENCODER = "cross-encoder"
    LEXICAL = "lexical"


class LLMProvider(StrEnum):
    OLLAMA = "ollama"
    FAKE = "fake"  # deterministic extractive answers; tests & offline demos


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_ignore_empty=True,  # `MIN_RELEVANCE_SCORE=` means "use the default"
    )

    # --- Application -----------------------------------------------------------------
    app_name: str = "DocIntel - AI Document Intelligence Platform"
    environment: str = Field(default="development", pattern="^(development|test|production)$")
    log_level: str = "INFO"
    log_json: bool = False
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # --- Storage -------------------------------------------------------------------
    database_url: str = "postgresql+psycopg://docintel:docintel@localhost:5432/docintel"
    database_echo: bool = False
    auto_create_schema: bool = True  # Alembic is the source of truth in production
    upload_dir: Path = Path("./data/uploads")
    max_upload_size_mb: int = Field(default=25, ge=1, le=500)
    allowed_extensions: list[str] = [".pdf", ".docx", ".txt"]

    # --- Chunking ------------------------------------------------------------------
    chunk_size_tokens: int = Field(default=128, ge=32, le=2048)
    chunk_overlap_tokens: int = Field(default=24, ge=0, le=512)
    min_chunk_tokens: int = Field(default=20, ge=1)

    # --- Embeddings ----------------------------------------------------------------
    embedding_provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMERS
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    embedding_batch_size: int = Field(default=32, ge=1)
    embedding_device: str | None = None  # "cpu", "cuda", "mps" or None for auto
    warmup_models: bool = True  # load embedder + reranker weights at start-up
    hashing_embedding_dim: int = 512

    # --- Vector store --------------------------------------------------------------
    vector_store: VectorStoreProvider = VectorStoreProvider.CHROMA_HTTP
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_persist_dir: Path = Path("./data/chroma")
    chroma_collection: str = "document_chunks"

    # --- Retrieval -----------------------------------------------------------------
    retrieval_top_k: int = Field(default=6, ge=1, le=20)
    retrieval_candidates: int = Field(default=30, ge=5, le=200)
    hybrid_semantic_weight: float = Field(default=1.0, ge=0.0)
    hybrid_lexical_weight: float = Field(default=1.0, ge=0.0)
    rrf_k: int = Field(default=60, ge=1)

    # --- Reranking -----------------------------------------------------------------
    reranker_provider: RerankerProvider = RerankerProvider.CROSS_ENCODER
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = Field(default=20, ge=1, le=100)

    # --- Guardrails ----------------------------------------------------------------
    # Chunks whose normalised rerank relevance (0..1) falls below this threshold are
    # dropped. If nothing survives, the system refuses *before* calling the LLM.
    # Unset -> the reranker's own calibrated default (see app/rag/reranker.py).
    min_relevance_score: float | None = Field(default=None, ge=0.0, le=1.0)
    # Second evidence signal: dense cosine similarity. Cross-encoders under-score
    # paraphrases ("vacation" vs "annual leave"), so a chunk also counts as
    # evidence when the bi-encoder finds it similar enough. Unset -> the embedding
    # model's calibrated default (0.5 for bge-small-en-v1.5). Re-calibrate with
    # `python -m evaluation.run --models real` when changing models.
    min_semantic_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    # Post-generation lexical grounding check (share of answer content words that
    # occur in the cited sources). Used for the confidence label, and - when
    # ``strict_grounding`` is enabled - to refuse weakly grounded answers.
    grounding_high_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    grounding_medium_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    strict_grounding: bool = False

    # --- LLM -----------------------------------------------------------------------
    llm_provider: LLMProvider = LLMProvider.OLLAMA
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_num_ctx: int = Field(default=8192, ge=512)
    # Reasoning ("thinking") models: false = skip the hidden reasoning pass (much
    # faster on CPU); unset = model default. Ignored by non-thinking models.
    ollama_think: bool | None = None
    llm_timeout_seconds: float = Field(default=180.0, gt=0)
    max_context_chars: int = Field(default=12_000, ge=1000)
    history_turns: int = Field(default=4, ge=0, le=20)
    query_rewrite_enabled: bool = True

    @field_validator("allowed_extensions")
    @classmethod
    def _normalise_extensions(cls, value: list[str]) -> list[str]:
        return [ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in value]

    @model_validator(mode="after")
    def _check_chunking(self) -> Settings:
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_SIZE_TOKENS")
        if self.rerank_candidates > self.retrieval_candidates:
            self.rerank_candidates = self.retrieval_candidates
        return self

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
