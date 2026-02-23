from __future__ import annotations

import re
from collections import defaultdict

from excerpts import clamp_window, merge_overlapping_ranges, read_lines, slice_lines
from reranker import Reranker
from indexer import ContextIndexer
from search import CodeSearcher


TOKEN_RE = re.compile(r"[A-Za-z0-9_]{2,}")


def _tokenize(query: str) -> list[str]:
    terms = [m.group(0).lower() for m in TOKEN_RE.finditer(query)]
    seen: set[str] = set()
    deduped: list[str] = []
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        deduped.append(term)
    return deduped


def _normalize_excerpt_for_response(text: str) -> str:
    lines = [line.rstrip() for line in str(text).splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    collapsed: list[str] = []
    blank_open = False
    for line in lines:
        if not line.strip():
            if blank_open:
                continue
            blank_open = True
            collapsed.append("")
            continue
        blank_open = False
        collapsed.append(line)
    return "\n".join(collapsed)


def _compact_snippet_excerpts(snippets: list[dict], *, max_results: int) -> tuple[list[dict], bool]:
    # Keep MCP payloads bounded to avoid "Large MCP response" warnings in clients.
    # Approx conversion: 1 token ~= 4 chars, so this target is ~1800-2500 tokens total excerpt text.
    per_snippet_budget = 900
    total_budget = max(1600, min(10_000, max_results * per_snippet_budget))
    remaining = total_budget
    compacted = False
    out: list[dict] = []

    for snippet in snippets:
        cloned = dict(snippet)
        excerpt = _normalize_excerpt_for_response(str(cloned.get("excerpt", "")))
        if len(excerpt) > per_snippet_budget:
            excerpt = excerpt[:per_snippet_budget].rstrip() + "\n...[truncated]"
            compacted = True
        if remaining <= 80:
            excerpt = ""
            compacted = True
        elif len(excerpt) > remaining:
            keep = max(80, remaining)
            excerpt = excerpt[:keep].rstrip() + "\n...[truncated]"
            compacted = True
        remaining -= len(excerpt)
        cloned["excerpt"] = excerpt
        out.append(cloned)
    return out, compacted


def run(
    indexer: ContextIndexer,
    searcher: CodeSearcher,
    reranker: Reranker,
    *,
    query: str,
    scope_paths: list[str] | None,
    max_results: int,
    max_excerpt_lines: int,
    include_tests: bool,
) -> dict:
    effective_scope_paths = scope_paths
    indexed_records = indexer.iter_indexed(scope_paths=effective_scope_paths, include_tests=include_tests)
    if not indexed_records and scope_paths:
        effective_scope_paths = None
        indexed_records = indexer.iter_indexed(scope_paths=None, include_tests=include_tests)
    indexed_paths = [record.path for record in indexed_records]
    planner = getattr(reranker, "plan_retrieval", None)
    if callable(planner):
        retrieval_plan = planner(
            query=query,
            available_paths=indexed_paths,
            max_paths=min(max(max_results * 3, 8), 36),
            max_terms=6,
            tool_name="context.relevant_code",
            hint_paths=list(scope_paths or []),
            hint_terms=_tokenize(query)[:6],
        )
    else:
        retrieval_plan = {
            "paths": [],
            "terms": _tokenize(query),
            "rationale": "Planner unavailable; deterministic term fallback.",
            "source": "deterministic_fallback",
        }

    terms = [term.lower() for term in retrieval_plan.get("terms", []) if isinstance(term, str) and term.strip()]
    if not terms:
        terms = _tokenize(query)
    if not terms:
        return {
            "snippets": [],
            "coverage": {
                "files_considered": 0,
                "candidates_scanned": 0,
                "truncated": False,
            },
        }

    densities, hit_lines, files_considered = searcher.term_density(
        terms=terms,
        scope_paths=effective_scope_paths,
        include_tests=include_tests,
    )

    preferred_paths = [
        path
        for path in retrieval_plan.get("paths", [])
        if isinstance(path, str) and path in indexed_paths
    ]
    preferred_rank = {path: idx for idx, path in enumerate(preferred_paths)}
    preferred_count = len(preferred_paths)

    for path in preferred_paths:
        if path not in densities:
            densities[path] = 0.0
        if path not in hit_lines:
            hit_lines[path] = [1]

    def ranking_score(item: tuple[str, float]) -> float:
        path, density = item
        if path in preferred_rank:
            # Gemini-selected paths get precedence, density refines tie-break ordering.
            return float(preferred_count - preferred_rank[path]) * 1000.0 + float(density)
        return float(density)

    ranked_files = sorted(densities.items(), key=ranking_score, reverse=True)
    candidate_limit = max(max_results * 6, 12)
    adaptive_excerpt_lines = max(24, min(120, int(240 / max(1, max_results))))
    effective_excerpt_lines = min(max_excerpt_lines, adaptive_excerpt_lines)
    window_lines = effective_excerpt_lines

    candidates: list[dict] = []
    candidates_by_file: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for path, density in ranked_files:
        if len(candidates) >= candidate_limit:
            break

        abs_path = indexer.abs_path(path)
        lines = read_lines(abs_path)
        total_lines = len(lines)
        if total_lines == 0:
            continue

        raw_ranges = []
        for line_no in sorted(set(hit_lines.get(path, []))):
            raw_ranges.append(clamp_window(line_no, total_lines, window_lines))

        merged = merge_overlapping_ranges(raw_ranges)
        candidates_by_file[path].extend(merged)

        for start, end in merged:
            if len(candidates) >= candidate_limit:
                break
            excerpt = slice_lines(abs_path, start_line=start, end_line=end, max_lines=effective_excerpt_lines)
            snippet_hit_count = sum(start <= ln <= end for ln in hit_lines.get(path, []))
            base_score = float(density) + (snippet_hit_count / max(1, end - start + 1))
            candidates.append(
                {
                    "id": len(candidates),
                    "path": path,
                    "start_line": excerpt.start_line,
                    "end_line": excerpt.end_line,
                    "excerpt": excerpt.excerpt,
                    "base_score": base_score,
                    "hit_count": snippet_hit_count,
                }
            )

    truncated = len(ranked_files) > 0 and len(candidates) >= candidate_limit
    reranked = reranker.rerank(query=query, candidates=candidates, max_results=max_results)

    candidate_map = {c["id"]: c for c in candidates}
    snippets: list[dict] = []

    for item in reranked:
        candidate = candidate_map.get(item["id"])
        if not candidate:
            continue
        abs_path = indexer.abs_path(candidate["path"])
        source_slice = slice_lines(
            abs_path,
            start_line=candidate["start_line"],
            end_line=candidate["end_line"],
            max_lines=effective_excerpt_lines,
        )
        snippets.append(
            {
                "path": candidate["path"],
                "start_line": source_slice.start_line,
                "end_line": source_slice.end_line,
                "excerpt": source_slice.excerpt,
                "score": float(item["score"]),
                "rationale": item["rationale"],
            }
        )

    snippets.sort(key=lambda s: s["score"], reverse=True)
    compacted_snippets, response_compacted = _compact_snippet_excerpts(
        snippets[:max_results],
        max_results=max_results,
    )

    return {
        "snippets": compacted_snippets,
        "coverage": {
            "files_considered": files_considered,
            "candidates_scanned": len(candidates),
            "truncated": truncated,
            "response_compacted": response_compacted,
            "effective_excerpt_lines": effective_excerpt_lines,
        },
        "retrieval_plan": {
            "source": retrieval_plan.get("source", "deterministic_fallback"),
            "paths_selected": len(preferred_paths),
            "terms_selected": terms[:12],
            "rationale": retrieval_plan.get("rationale", ""),
        },
    }
