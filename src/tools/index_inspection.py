from __future__ import annotations

from indexer import ContextIndexer


def run(
    indexer: ContextIndexer,
    *,
    max_file_bytes: int,
    max_excerpt_lines: int,
    gemini_timeout_seconds: int,
    gemini_status: dict,
) -> dict:
    stats = indexer.stats()
    return {
        "roots": stats["roots"],
        "last_scan": stats["last_scan"],
        "files_indexed": stats["files_indexed"],
        "files_ignored": stats["files_ignored"],
        "providers": ["ripgrep", "watchdog", "excerpt-extractor"],
        "limits": {
            "max_file_bytes": max_file_bytes,
            "max_excerpt_lines": max_excerpt_lines,
            "gemini_timeout_seconds": gemini_timeout_seconds,
        },
        "gemini": gemini_status,
    }
