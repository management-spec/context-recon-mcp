from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

DEFAULT_IGNORES = [
    ".git/",
    "node_modules/",
    ".venv/",
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
]
SYSTEM_TOP_LEVEL_DIRS = {
    "Applications",
    "Library",
    "System",
    "Volumes",
    "cores",
    "dev",
    "private",
    "usr",
    "opt",
    "etc",
}


@dataclass
class IgnoreMatcher:
    root: Path
    extra_ignores: list[str] | None = None

    def __post_init__(self) -> None:
        self.root = self.root.resolve()
        self.patterns = self._load_gitignore_patterns()
        self.patterns.extend(DEFAULT_IGNORES)
        if self.extra_ignores:
            self.patterns.extend(self.extra_ignores)

    def _load_gitignore_patterns(self) -> list[str]:
        gitignore = self.root / ".gitignore"
        if not gitignore.exists():
            return []

        patterns: list[str] = []
        for line in gitignore.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            patterns.append(line)
        return patterns

    def _match_single(self, rel_path: str, pattern: str, is_dir: bool) -> bool:
        anchored = pattern.startswith("/")
        if anchored:
            pattern = pattern[1:]

        dir_pattern = pattern.endswith("/")
        if dir_pattern:
            pattern = pattern[:-1]

        rel = rel_path
        name = rel.split("/")[-1]

        if anchored:
            if dir_pattern:
                return rel == pattern or rel.startswith(f"{pattern}/")
            return fnmatch(rel, pattern)

        if "/" in pattern:
            if dir_pattern:
                return rel == pattern or rel.startswith(f"{pattern}/")
            return fnmatch(rel, pattern)

        # Basename-only pattern applies anywhere in tree.
        if dir_pattern:
            return pattern in rel.split("/")

        return fnmatch(name, pattern) or fnmatch(rel, f"**/{pattern}")

    def is_ignored(self, path: Path) -> bool:
        try:
            rel = path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return True

        if not rel:
            return False

        parts = rel.split("/")
        # Hide all dot-prefixed files/folders by default across tool surfaces.
        if any(part.startswith(".") for part in parts):
            return True
        if "__MACOSX" in parts:
            return True
        if any(part in {".git", "node_modules", ".venv"} for part in parts):
            return True
        if self._is_system_folder_path(parts):
            return True

        ignored = False
        is_dir = path.is_dir()

        for raw in self.patterns:
            negate = raw.startswith("!")
            pattern = raw[1:] if negate else raw
            if self._match_single(rel, pattern, is_dir):
                ignored = not negate

        return ignored

    def _is_system_folder_path(self, parts: list[str]) -> bool:
        if not parts:
            return False
        # Only apply host-system top-level folder filtering when root is broad
        # (home or filesystem root). Project roots should not be over-filtered.
        if self.root not in {Path("/"), Path.home().resolve()}:
            return False
        return parts[0] in SYSTEM_TOP_LEVEL_DIRS

    def effective_ignores(self) -> list[str]:
        return sorted(set(self.patterns))
