from __future__ import annotations

import re

from gemini import GeminiReranker
from indexer import ContextIndexer
from search import CodeSearcher

MAX_HIT_TEXT_CHARS = 220
MAX_HITS_TEXT_BUDGET = 8_000


def _tokenize_pattern(pattern: str, max_terms: int = 4) -> list[str]:
    terms = re.findall(r"[A-Za-z0-9_]{2,}", pattern.lower())
    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        deduped.append(term)
        if len(deduped) >= max_terms:
            break
    return deduped


def run(
    searcher: CodeSearcher,
    indexer: ContextIndexer,
    reranker: GeminiReranker,
    *,
    pattern: str,
    regex: bool,
    case_sensitive: bool,
    scope_paths: list[str] | None,
    max_hits: int,
) -> dict:
    effective_scope_paths = scope_paths
    indexed_records = indexer.iter_indexed(scope_paths=effective_scope_paths, include_tests=True)
    if not indexed_records and scope_paths:
        effective_scope_paths = None
        indexed_records = indexer.iter_indexed(scope_paths=None, include_tests=True)
    indexed_paths = [record.path for record in indexed_records]
    planner = getattr(reranker, "plan_retrieval", None)
    if callable(planner):
        hint_terms = _tokenize_pattern(pattern, max_terms=4)
        if not hint_terms and pattern.strip():
            hint_terms = [pattern.strip()]
        retrieval_plan = planner(
            query=pattern,
            available_paths=indexed_paths,
            max_paths=min(max(max_hits, 8), 32),
            max_terms=4,
            tool_name="context.code_search",
            hint_paths=list(scope_paths or []),
            hint_terms=hint_terms,
        )
    else:
        retrieval_plan = {
            "paths": [],
            "terms": [pattern],
            "rationale": "Gemini planning unavailable; deterministic pattern fallback.",
            "source": "deterministic_fallback",
        }
    planned_paths = {
        path
        for path in retrieval_plan.get("paths", [])
        if isinstance(path, str)
    }
    planned_terms = [
        term
        for term in retrieval_plan.get("terms", [])
        if isinstance(term, str) and term.strip()
    ]

    patterns: list[tuple[str, bool]] = []
    if regex:
        patterns.append((pattern, True))
    else:
        if planned_terms:
            for term in planned_terms:
                patterns.append((term, False))
        if pattern not in [item[0] for item in patterns]:
            patterns.append((pattern, False))

    merged: dict[tuple[str, int], dict] = {}
    for pattern_value, pattern_regex in patterns:
        hits = searcher.code_search(
            pattern=pattern_value,
            regex=pattern_regex,
            case_sensitive=case_sensitive,
            scope_paths=effective_scope_paths,
            max_hits=max(max_hits * 3, 60),
        )
        for hit in hits:
            if planned_paths and hit.path not in planned_paths:
                continue
            key = (hit.path, hit.line)
            if key in merged:
                continue
            merged[key] = {"path": hit.path, "line": hit.line, "text": hit.text}
            if len(merged) >= max_hits:
                break
        if len(merged) >= max_hits:
            break

    if not merged:
        fallback_hits = searcher.code_search(
            pattern=pattern,
            regex=regex,
            case_sensitive=case_sensitive,
            scope_paths=effective_scope_paths,
            max_hits=max_hits,
        )
        merged = {(hit.path, hit.line): {"path": hit.path, "line": hit.line, "text": hit.text} for hit in fallback_hits}

    path_rank = {
        path: idx
        for idx, path in enumerate(retrieval_plan.get("paths", []) if isinstance(retrieval_plan.get("paths", []), list) else [])
        if isinstance(path, str)
    }
    hits_sorted = sorted(
        merged.values(),
        key=lambda item: (path_rank.get(item["path"], 10_000), item["path"], int(item["line"])),
    )

    response_compacted = False
    remaining_budget = MAX_HITS_TEXT_BUDGET
    compacted_hits: list[dict] = []
    for hit in hits_sorted[:max_hits]:
        text = str(hit.get("text", "")).replace("\t", " ").rstrip()
        if len(text) > MAX_HIT_TEXT_CHARS:
            text = text[:MAX_HIT_TEXT_CHARS].rstrip() + "...[truncated]"
            response_compacted = True
        if remaining_budget <= 0:
            response_compacted = True
            text = ""
        elif len(text) > remaining_budget:
            keep = max(40, remaining_budget)
            text = text[:keep].rstrip() + "...[truncated]"
            response_compacted = True
        remaining_budget -= len(text)
        compacted_hits.append(
            {
                "path": hit["path"],
                "line": int(hit["line"]),
                "text": text,
            }
        )

    return {
        "hits": compacted_hits,
        "retrieval_plan": {
            "source": retrieval_plan.get("source", "deterministic_fallback"),
            "paths_selected": len(planned_paths),
            "terms_selected": planned_terms[:6],
            "rationale": retrieval_plan.get("rationale", ""),
        },
        "response_compacted": response_compacted,
    }
