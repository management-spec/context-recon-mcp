from __future__ import annotations

from reranker import Reranker
from indexer import ContextIndexer
from search import CodeSearcher
from tools import relevant_code


def run(
    *,
    indexer: ContextIndexer,
    searcher: CodeSearcher,
    reranker: Reranker,
    query: str | None,
    question: str | None,
    scope_paths: list[str] | None,
    max_results: int,
    max_excerpt_lines: int,
    include_tests: bool,
) -> dict:
    resolved_query = (query or question or "").strip()
    if not resolved_query:
        raise ValueError("Provide either `query` or `question`.")

    result = relevant_code.run(
        indexer=indexer,
        searcher=searcher,
        reranker=reranker,
        query=resolved_query,
        scope_paths=scope_paths,
        max_results=max_results,
        max_excerpt_lines=max_excerpt_lines,
        include_tests=include_tests,
    )

    return {
        "snippets": result["snippets"],
        "coverage": result["coverage"],
        "retrieval_plan": result.get("retrieval_plan", {}),
        "retrieval_mode": "context_recon",
    }
