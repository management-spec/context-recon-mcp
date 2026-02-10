from __future__ import annotations

from pathlib import Path

from indexer import ContextIndexer

KEY_FILE_REASONS = {
    "pyproject.toml": "Python project metadata and dependencies",
    "README.md": "Project usage and architecture overview",
    "config.yaml": "Context engine runtime configuration",
    "src/server.py": "MCP server entrypoint",
}
MAX_TREE_LINES = 450
MAX_TREE_CHARS = 12_000


def _render_tree(
    root: Path,
    max_depth: int,
    *,
    prefix: str = "",
    indexer: ContextIndexer,
) -> list[str]:
    if max_depth < 0 or not root.exists() or not root.is_dir():
        return []

    lines: list[str] = []
    entries = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    for entry in entries:
        # Hide dot-prefixed files/folders from overview to reduce system/noise output.
        if entry.name.startswith("."):
            continue
        if indexer.ignore_matcher.is_ignored(entry):
            continue
        marker = "/" if entry.is_dir() else ""
        lines.append(f"{prefix}{entry.name}{marker}")
        if entry.is_dir() and max_depth > 0:
            lines.extend(
                _render_tree(
                    entry,
                    max_depth - 1,
                    prefix=prefix + "  ",
                    indexer=indexer,
                )
            )
    return lines


def run(
    indexer: ContextIndexer,
    scope_paths: list[str] | None,
    max_depth: int,
) -> dict:
    scopes = scope_paths or [root.relative_to(indexer.workspace_root).as_posix() for root in indexer.roots]
    tree_lines: list[str] = []

    for scope in scopes:
        base = (indexer.workspace_root / scope).resolve()
        if not base.exists():
            continue
        if indexer.ignore_matcher.is_ignored(base):
            continue
        title = scope if scope else "."
        tree_lines.append(f"{title}/")
        tree_lines.extend(_render_tree(base, max_depth=max_depth, indexer=indexer))

    key_files = []
    for rel in indexer.existing_paths():
        if rel in KEY_FILE_REASONS:
            key_files.append({"path": rel, "reason": KEY_FILE_REASONS[rel]})
        elif rel.endswith("server.py"):
            key_files.append({"path": rel, "reason": "MCP entrypoint"})

    limited_lines = tree_lines[:MAX_TREE_LINES]
    tree = "\n".join(limited_lines)
    tree_truncated = len(tree_lines) > MAX_TREE_LINES
    if len(tree) > MAX_TREE_CHARS:
        tree = tree[:MAX_TREE_CHARS].rstrip() + "\n...[truncated]"
        tree_truncated = True

    return {
        "tree": tree,
        "key_files": key_files[:30],
        "ignores": indexer.ignore_matcher.effective_ignores(),
        "tree_truncated": tree_truncated,
    }
