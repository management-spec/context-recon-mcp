from __future__ import annotations

import re
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from indexer import ContextIndexer


@dataclass(frozen=True)
class SearchHit:
    path: str
    line: int
    text: str


class CodeSearcher:
    def __init__(self, indexer: ContextIndexer) -> None:
        self.indexer = indexer

    def _is_allowed_path(self, abs_path: Path) -> bool:
        return not self.indexer.ignore_matcher.is_ignored(abs_path)

    def _resolve_scope_paths(self, scope_paths: list[str] | None) -> list[Path]:
        if not scope_paths:
            return self.indexer.roots

        resolved: list[Path] = []
        for scope in scope_paths:
            candidate = (self.indexer.workspace_root / scope).resolve()
            if not self._is_allowed_path(candidate):
                continue
            if any(root == candidate or root in candidate.parents for root in self.indexer.roots):
                resolved.append(candidate)

        return resolved or self.indexer.roots

    def _use_ripgrep(self) -> bool:
        return shutil.which("rg") is not None

    def _ripgrep_search(
        self,
        pattern: str,
        regex: bool,
        case_sensitive: bool,
        scope_paths: list[str] | None,
        max_hits: int,
    ) -> list[SearchHit]:
        args = ["rg", "--line-number", "--no-heading", "--color", "never"]
        if not regex:
            args.append("-F")
        if not case_sensitive:
            args.append("-i")

        args.extend(["--max-count", str(max_hits), pattern])
        args.extend([str(path) for path in self._resolve_scope_paths(scope_paths)])

        proc = subprocess.run(args, capture_output=True, text=True, check=False)
        if proc.returncode not in (0, 1):
            return []

        hits: list[SearchHit] = []
        for line in proc.stdout.splitlines():
            # rg output format: path:line:text
            parts = line.split(":", 2)
            if len(parts) != 3:
                continue
            raw_path, raw_line, text = parts
            try:
                abs_path = Path(raw_path).resolve()
                rel_path = abs_path.relative_to(self.indexer.workspace_root).as_posix()
                line_num = int(raw_line)
            except (ValueError, OSError):
                continue
            if not self._is_allowed_path(abs_path):
                continue

            hits.append(SearchHit(path=rel_path, line=line_num, text=text))
            if len(hits) >= max_hits:
                break

        return hits

    def _scan_search(
        self,
        pattern: str,
        regex: bool,
        case_sensitive: bool,
        scope_paths: list[str] | None,
        max_hits: int,
    ) -> list[SearchHit]:
        flags = 0 if case_sensitive else re.IGNORECASE
        expr = re.compile(pattern if regex else re.escape(pattern), flags)

        hits: list[SearchHit] = []
        records = self.indexer.iter_indexed(scope_paths=scope_paths, include_tests=True)
        for record in records:
            abs_path = self.indexer.abs_path(record.path)
            try:
                lines = abs_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for idx, line in enumerate(lines, start=1):
                if expr.search(line):
                    hits.append(SearchHit(path=record.path, line=idx, text=line))
                    if len(hits) >= max_hits:
                        return hits

        return hits

    def code_search(
        self,
        pattern: str,
        regex: bool,
        case_sensitive: bool,
        scope_paths: list[str] | None,
        max_hits: int,
    ) -> list[SearchHit]:
        if self._use_ripgrep():
            hits = self._ripgrep_search(pattern, regex, case_sensitive, scope_paths, max_hits)
            if hits:
                return hits
        return self._scan_search(pattern, regex, case_sensitive, scope_paths, max_hits)

    def term_density(
        self,
        terms: list[str],
        scope_paths: list[str] | None,
        include_tests: bool,
    ) -> tuple[dict[str, float], dict[str, list[int]], int]:
        term_hits: dict[str, list[int]] = defaultdict(list)
        total_considered = 0

        records = self.indexer.iter_indexed(scope_paths=scope_paths, include_tests=include_tests)
        total_considered = len(records)

        if self._use_ripgrep():
            for term in terms:
                args = [
                    "rg",
                    "--line-number",
                    "--no-heading",
                    "--color",
                    "never",
                    "-i",
                    "-F",
                    term,
                ]
                args.extend([str(path) for path in self._resolve_scope_paths(scope_paths)])
                proc = subprocess.run(args, capture_output=True, text=True, check=False)
                if proc.returncode not in (0, 1):
                    continue
                for line in proc.stdout.splitlines():
                    parts = line.split(":", 2)
                    if len(parts) != 3:
                        continue
                    raw_path, raw_line, _ = parts
                    try:
                        abs_path = Path(raw_path).resolve()
                        rel_path = abs_path.relative_to(self.indexer.workspace_root).as_posix()
                        line_num = int(raw_line)
                    except (ValueError, OSError):
                        continue
                    if not self._is_allowed_path(abs_path):
                        continue
                    term_hits[rel_path].append(line_num)

            if term_hits:
                densities: dict[str, float] = {}
                for path, lines in term_hits.items():
                    record = next((r for r in records if r.path == path), None)
                    if record is None or record.size <= 0:
                        continue
                    densities[path] = len(lines) / max(record.size, 1)
                return densities, term_hits, total_considered

        # fallback scanner
        lowered_terms = [term.lower() for term in terms if term]
        for record in records:
            abs_path = self.indexer.abs_path(record.path)
            try:
                lines = abs_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line_no, line in enumerate(lines, start=1):
                lower_line = line.lower()
                if any(term in lower_line for term in lowered_terms):
                    term_hits[record.path].append(line_no)

        densities = {
            path: len(lines) / max(next((r.size for r in records if r.path == path), 1), 1)
            for path, lines in term_hits.items()
        }
        return densities, term_hits, total_considered
