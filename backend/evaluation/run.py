"""CLI: python -m evaluation.run [--models real] [--llm ollama] [--out reports/]

Examples::

    # Fast, dependency-free (hashing embedder, lexical reranker, fake LLM)
    python -m evaluation.run

    # Real retrieval stack (downloads bge-small + MiniLM cross-encoder)
    python -m evaluation.run --models real --no-answers

    # Full end-to-end with a local LLM (requires `ollama serve` + model pulled)
    python -m evaluation.run --models real --llm ollama --ollama-model llama3.2:3b
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from app.core.config import EmbeddingProvider, LLMProvider, RerankerProvider
from evaluation.harness import EvalConfig, run_evaluation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DocIntel retrieval & answer evaluation")
    parser.add_argument("--models", choices=["fake", "real"], default="fake")
    parser.add_argument("--llm", choices=["fake", "ollama"], default="fake")
    parser.add_argument("--ollama-model", default=None)
    parser.add_argument("--min-relevance", type=float, default=None)
    parser.add_argument("--no-answers", action="store_true", help="retrieval metrics only")
    parser.add_argument("--out", type=Path, default=Path("evaluation/reports"))
    args = parser.parse_args(argv)

    real = args.models == "real"
    config = EvalConfig(
        embedding_provider=EmbeddingProvider.SENTENCE_TRANSFORMERS
        if real
        else EmbeddingProvider.HASHING,
        reranker_provider=RerankerProvider.CROSS_ENCODER if real else RerankerProvider.LEXICAL,
        llm_provider=LLMProvider.OLLAMA if args.llm == "ollama" else LLMProvider.FAKE,
        ollama_model=args.ollama_model,
        min_relevance=args.min_relevance,
    )
    report = run_evaluation(config, answers=not args.no_answers)
    markdown = report.to_markdown()
    print(markdown)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"eval-{args.models}-{args.llm}"
    (args.out / f"{stem}.md").write_text(markdown, encoding="utf-8")
    (args.out / f"{stem}.json").write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print(f"Report written to {args.out / stem}.md/.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
