from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{2,}")


def _tokenize(text: str, *, max_terms: int) -> list[str]:
    terms = [match.group(0).lower() for match in _TOKEN_RE.finditer(text or "")]
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


def _truncate(text: str, *, max_chars: int) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max(0, max_chars - 3)].rstrip() + "..."


def build_factual_rerank_rationale(
    *,
    query: str,
    excerpt: str,
    path: str,
    max_chars: int = 180,
) -> str:
    query_terms = _tokenize(query, max_terms=16)
    excerpt_terms = set(_tokenize(excerpt, max_terms=256))
    path_lower = (path or "").lower()

    evidence: list[str] = []
    for term in query_terms:
        if term in excerpt_terms:
            evidence.append(term)
        if len(evidence) >= 4:
            break
    if len(evidence) < 4:
        for term in query_terms:
            if term in evidence:
                continue
            if term in path_lower:
                evidence.append(term)
            if len(evidence) >= 4:
                break

    if evidence:
        return _truncate(
            f"Matched query terms in local code context: {', '.join(evidence)}.",
            max_chars=max_chars,
        )
    return _truncate(
        "No direct query-term overlap found in this excerpt; selection is based on local retrieval signals.",
        max_chars=max_chars,
    )


def build_factual_plan_rationale(
    *,
    selected_paths: list[str],
    selected_terms: list[str],
    max_chars: int = 180,
) -> str:
    path_count = len(selected_paths)
    term_count = len(selected_terms)
    if path_count and term_count:
        return _truncate(
            f"Selected {path_count} indexed path(s) and {term_count} query term(s) from the local catalog.",
            max_chars=max_chars,
        )
    if path_count:
        return _truncate(
            f"Selected {path_count} indexed path(s) from the local catalog.",
            max_chars=max_chars,
        )
    if term_count:
        return _truncate(
            f"Selected {term_count} query term(s) from the local query and hints.",
            max_chars=max_chars,
        )
    return "No indexed paths or terms were selected."
