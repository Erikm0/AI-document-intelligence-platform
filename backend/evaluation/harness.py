"""Evaluation harness shared by the CLI (``python -m evaluation.run``) and pytest.

It builds the synthetic corpus, ingests it through the *real* ingestion pipeline
and then measures:

* **Retrieval quality** per mode (semantic / lexical / hybrid / hybrid+rerank):
  Recall@1/3/5, MRR, nDCG@5 and page accuracy.
* **Answer quality / hallucination resistance**: answer rate on answerable
  questions, refusal rate on unanswerable ones, citation accuracy (does a cited
  source contain the evidence?) and keyword accuracy.
"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.container import Container, build_container
from app.core.config import (
    EmbeddingProvider,
    LLMProvider,
    RerankerProvider,
    Settings,
    VectorStoreProvider,
)
from app.db.models import Document, DocumentStatus
from app.ingestion.parsers import CONTENT_TYPES, EXTENSION_TO_TYPE
from app.rag.llm import LLMClient
from app.rag.types import RetrievedChunk
from app.schemas.chat import RetrievalMode
from evaluation.corpus import build_corpus
from evaluation.metrics import mean, ndcg_at_k, recall_at_k, reciprocal_rank

DATASET_PATH = Path(__file__).with_name("dataset.json")


def load_dataset(path: Path = DATASET_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class EvalConfig:
    embedding_provider: EmbeddingProvider = EmbeddingProvider.HASHING
    reranker_provider: RerankerProvider = RerankerProvider.LEXICAL
    llm_provider: LLMProvider = LLMProvider.FAKE
    ollama_model: str | None = None
    min_relevance: float | None = None
    top_k: int = 5


@dataclass
class EvalReport:
    config: dict[str, Any]
    retrieval: dict[str, dict[str, float]] = field(default_factory=dict)
    answers: dict[str, float] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)
    ingestion: dict[str, Any] = field(default_factory=dict)

    def to_markdown(self) -> str:
        lines = ["# DocIntel evaluation report", ""]
        lines += [f"- **{k}**: `{v}`" for k, v in self.config.items()]
        lines += ["", "## Ingestion", ""]
        lines += [f"- **{k}**: {v}" for k, v in self.ingestion.items()]
        lines += ["", "## Retrieval", ""]
        if self.retrieval:
            metrics = list(next(iter(self.retrieval.values())).keys())
            lines.append("| mode | " + " | ".join(metrics) + " |")
            lines.append("|---|" + "---|" * len(metrics))
            for mode, values in self.retrieval.items():
                lines.append(
                    f"| {mode} | " + " | ".join(f"{values[m]:.3f}" for m in metrics) + " |"
                )
        if self.answers:
            lines += [
                "",
                "## Answers & hallucination resistance",
                "",
                "| metric | value |",
                "|---|---|",
            ]
            lines += [f"| {k} | {v:.3f} |" for k, v in self.answers.items()]
        if self.failures:
            lines += ["", "## Failures", ""]
            lines += [f"- `{f['id']}` ({f['kind']}): {f['detail']}" for f in self.failures[:50]]
        return "\n".join(lines) + "\n"


def build_eval_container(
    config: EvalConfig, workdir: Path, *, llm: LLMClient | None = None
) -> Container:
    overrides: dict[str, Any] = {
        "environment": "test",
        "database_url": f"sqlite:///{(workdir / 'eval.db').as_posix()}",
        "upload_dir": workdir / "uploads",
        "vector_store": VectorStoreProvider.MEMORY,
        "embedding_provider": config.embedding_provider,
        "reranker_provider": config.reranker_provider,
        "llm_provider": config.llm_provider,
        "retrieval_top_k": config.top_k,
        "min_relevance_score": config.min_relevance,
        "log_level": "WARNING",
    }
    if config.ollama_model:
        overrides["ollama_model"] = config.ollama_model
    settings = Settings(**overrides)
    container = build_container(settings, llm=llm, synchronous_ingestion=True)
    container.startup()
    return container


def ingest_corpus(container: Container, corpus_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    files = build_corpus(corpus_dir)
    ids = []
    for path in files:
        upload = container.storage.receive(path.open("rb"), EXTENSION_TO_TYPE[path.suffix])
        with container.db.session() as session:
            file_type = EXTENSION_TO_TYPE[path.suffix]
            document = Document(
                filename=path.name,
                file_type=file_type,
                content_type=CONTENT_TYPES[file_type],
                size_bytes=upload.size_bytes,
                sha256=upload.sha256,
                storage_path="",
                status=DocumentStatus.PENDING,
                metadata_={},
            )
            session.add(document)
            session.flush()
            document.storage_path = str(container.storage.commit(upload, document.id, path.suffix))
            ids.append(document.id)
        container.ingestion.process(document.id)

    chunks = pages = 0
    with container.db.session() as session:
        for doc_id in ids:
            document = session.get(Document, doc_id)
            assert document is not None
            if document.status != DocumentStatus.READY:
                raise RuntimeError(
                    f"Ingestion failed for {document.filename}: {document.error_message}"
                )
            chunks += document.chunk_count
            pages += document.page_count or 0
    return {
        "documents": len(ids),
        "pages": pages,
        "chunks": chunks,
        "ingest_seconds": round(time.perf_counter() - started, 2),
    }


def _is_relevant(chunk: RetrievedChunk, item: dict[str, Any]) -> bool:
    return chunk.filename == item["doc"] and item["evidence"].lower() in chunk.text.lower()


def evaluate_retrieval(container: Container, dataset: dict[str, Any], report: EvalReport) -> None:
    modes = {
        "semantic": (RetrievalMode.SEMANTIC, False),
        "lexical (BM25)": (RetrievalMode.LEXICAL, False),
        "hybrid (RRF)": (RetrievalMode.HYBRID, False),
        "hybrid + rerank": (RetrievalMode.HYBRID, True),
    }
    for label, (mode, rerank) in modes.items():
        r1, r3, r5, mrr, ndcg, page_acc, latency = [], [], [], [], [], [], []
        for item in dataset["questions"]:
            t0 = time.perf_counter()
            result = container.retriever.retrieve(
                item["question"], top_k=10, mode=mode, rerank=rerank
            )
            latency.append((time.perf_counter() - t0) * 1000)
            flags = [_is_relevant(c, item) for c in result.chunks]
            r1.append(recall_at_k(flags, 1))
            r3.append(recall_at_k(flags, 3))
            r5.append(recall_at_k(flags, 5))
            mrr.append(reciprocal_rank(flags))
            ndcg.append(ndcg_at_k(flags, 5))
            first = next((c for c, f in zip(result.chunks, flags, strict=False) if f), None)
            page_ok = first is not None and (
                first.page_start is not None
                and first.page_start <= item["page"] <= (first.page_end or first.page_start)
            )
            page_acc.append(1.0 if page_ok else 0.0)
            if rerank and not flags[:5].count(True):
                report.failures.append(
                    {"id": item["id"], "kind": "retrieval_miss@5", "detail": item["question"]}
                )
        report.retrieval[label] = {
            "recall@1": mean(r1),
            "recall@3": mean(r3),
            "recall@5": mean(r5),
            "MRR": mean(mrr),
            "nDCG@5": mean(ndcg),
            "page_acc": mean(page_acc),
            "latency_ms": mean(latency),
        }


def evaluate_answers(container: Container, dataset: dict[str, Any], report: EvalReport) -> None:
    pipeline = container.pipeline
    answered = cited_ok = keyword_ok = 0
    grounding: list[float] = []
    for item in dataset["questions"]:
        result = pipeline.answer(item["question"])
        if result.refused:
            report.failures.append(
                {"id": item["id"], "kind": "false_refusal", "detail": result.refusal_reason}
            )
            continue
        answered += 1
        grounding.append(result.grounding_score)
        cited = [s.chunk for s in result.sources if s.cited]
        if any(_is_relevant(c, item) for c in cited):
            cited_ok += 1
        else:
            report.failures.append(
                {"id": item["id"], "kind": "wrong_citation", "detail": result.answer[:160]}
            )
        if all(k.lower() in result.answer.lower() for k in item.get("answer_keywords", [])):
            keyword_ok += 1

    refused = 0
    for item in dataset["unanswerable"]:
        result = pipeline.answer(item["question"])
        if result.refused:
            refused += 1
        else:
            report.failures.append(
                {"id": item["id"], "kind": "hallucination", "detail": result.answer[:200]}
            )

    n_ans = len(dataset["questions"])
    n_un = len(dataset["unanswerable"])
    report.answers = {
        "answer_rate (answerable)": answered / n_ans,
        "citation_accuracy (answered)": cited_ok / answered if answered else 0.0,
        "keyword_accuracy (answered)": keyword_ok / answered if answered else 0.0,
        "mean_grounding_score": mean(grounding),
        "refusal_rate (unanswerable)": refused / n_un,
        "hallucination_rate (unanswerable)": 1 - refused / n_un,
    }


def run_evaluation(
    config: EvalConfig,
    *,
    answers: bool = True,
    llm: LLMClient | None = None,
    workdir: Path | None = None,
) -> EvalReport:
    dataset = load_dataset()
    tmp = None
    if workdir is None:
        tmp = tempfile.TemporaryDirectory(prefix="docintel-eval-")
        workdir = Path(tmp.name)
    try:
        container = build_eval_container(config, workdir, llm=llm)
        try:
            report = EvalReport(
                config={
                    "embedder": container.embedder.name,
                    "reranker": container.reranker.name,
                    "llm": container.llm.model,
                    "min_relevance": container.pipeline.config.min_relevance,
                    "questions": len(dataset["questions"]),
                    "unanswerable": len(dataset["unanswerable"]),
                }
            )
            report.ingestion = ingest_corpus(container, workdir / "corpus")
            evaluate_retrieval(container, dataset, report)
            if answers:
                evaluate_answers(container, dataset, report)
            return report
        finally:
            container.shutdown()
    finally:
        if tmp is not None:
            tmp.cleanup()
