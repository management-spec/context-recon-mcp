from __future__ import annotations

from datetime import datetime, timezone

from excerpts import slice_lines
from indexer import ContextIndexer
from utils.hashing import sha256_file

MAX_SLICE_RESPONSE_CHARS = 8_000


def _normalize_excerpt(text: str) -> str:
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


def run(
    indexer: ContextIndexer,
    *,
    path: str,
    start_line: int,
    end_line: int,
    max_excerpt_lines: int = 200,
) -> dict:
    abs_path = indexer.abs_path(path)
    if indexer.ignore_matcher.is_ignored(abs_path):
        raise PermissionError(f"path is ignored by policy: {path}")
    if not abs_path.exists() or not abs_path.is_file():
        raise FileNotFoundError(path)

    excerpt = slice_lines(abs_path, start_line=start_line, end_line=end_line, max_lines=max_excerpt_lines)
    stat = abs_path.stat()
    response_compacted = False
    excerpt_text = _normalize_excerpt(excerpt.excerpt)
    if len(excerpt_text) > MAX_SLICE_RESPONSE_CHARS:
        excerpt_text = excerpt_text[:MAX_SLICE_RESPONSE_CHARS].rstrip() + "\n...[truncated]"
        response_compacted = True

    return {
        "path": path,
        "start_line": excerpt.start_line,
        "end_line": excerpt.end_line,
        "excerpt": excerpt_text,
        "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "size_bytes": int(stat.st_size),
        "token_estimate": max(0, len(excerpt_text) // 4),
        "sha256": sha256_file(abs_path),
        "response_compacted": response_compacted,
        "effective_excerpt_lines": int(max_excerpt_lines),
    }
