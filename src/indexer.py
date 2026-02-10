from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:
    from watchdog.events import FileSystemEvent, FileSystemEventHandler
    from watchdog.observers import Observer
    from watchdog.observers.polling import PollingObserver
except ImportError:  # pragma: no cover - exercised when dependency is installed.
    FileSystemEvent = object  # type: ignore[assignment]

    class FileSystemEventHandler:  # type: ignore[override]
        pass

    Observer = None  # type: ignore[assignment]
    PollingObserver = None  # type: ignore[assignment]

from utils.hashing import sha256_file
from utils.ignore import IgnoreMatcher

CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".kt",
    ".go",
    ".rs",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".swift",
    ".rb",
    ".php",
    ".scala",
    ".sql",
    ".sh",
    ".bash",
    ".zsh",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".xml",
    ".md",
}

DOC_EXTENSIONS = {".md", ".rst", ".txt", ".adoc"}
LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class FileRecord:
    path: str
    mtime: float
    size: int
    file_type: str


class _IndexUpdateHandler(FileSystemEventHandler):
    def __init__(self, indexer: "ContextIndexer") -> None:
        super().__init__()
        self.indexer = indexer

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self.indexer.try_index(Path(event.src_path))

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self.indexer.try_index(Path(event.src_path))

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self.indexer.remove(Path(event.src_path))
        self.indexer.try_index(Path(event.dest_path))

    def on_deleted(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self.indexer.remove(Path(event.src_path))


class ContextIndexer:
    def __init__(
        self,
        workspace_root: Path,
        roots: list[str],
        db_path: str,
        max_file_bytes: int,
        ignore_patterns: list[str] | None = None,
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.roots = [self._resolve_root(root) for root in roots]
        self.max_file_bytes = max_file_bytes
        self.ignore_matcher = IgnoreMatcher(self.workspace_root, ignore_patterns)
        raw_db_path = Path(db_path)
        if raw_db_path.is_absolute():
            self.db_path = raw_db_path.resolve()
        else:
            self.db_path = (self.workspace_root / raw_db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._observer: Observer | None = None
        self._init_db()

    def _resolve_root(self, root: str) -> Path:
        resolved = (self.workspace_root / root).resolve()
        return resolved

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS files (
                    path TEXT PRIMARY KEY,
                    mtime REAL NOT NULL,
                    size INTEGER NOT NULL,
                    file_type TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    indexed_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_usage (
                    tool TEXT NOT NULL,
                    query TEXT NOT NULL,
                    count INTEGER NOT NULL DEFAULT 0,
                    last_used TEXT NOT NULL,
                    PRIMARY KEY(tool, query)
                )
                """
            )

    def assert_writable(self) -> None:
        probe_key = "__context_engine_write_probe__"
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (probe_key, "ok"),
            )
            conn.execute("DELETE FROM meta WHERE key = ?", (probe_key,))

    def _set_meta(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def _get_meta(self, key: str, default: str = "") -> str:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def _is_binary(self, path: Path) -> bool:
        try:
            with path.open("rb") as handle:
                chunk = handle.read(2048)
                if b"\x00" in chunk:
                    return True
        except OSError:
            return True
        return False

    def _file_type(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in DOC_EXTENSIONS:
            return "doc"
        if suffix in CODE_EXTENSIONS:
            return "code"
        return "doc"

    def _is_within_roots(self, path: Path) -> bool:
        resolved = path.resolve()
        return any(root == resolved or root in resolved.parents for root in self.roots)

    def _to_rel(self, path: Path) -> str:
        return path.resolve().relative_to(self.workspace_root).as_posix()

    def _should_index(self, path: Path) -> bool:
        if not path.exists() or not path.is_file():
            return False
        if not self._is_within_roots(path):
            return False
        if self.ignore_matcher.is_ignored(path):
            return False
        try:
            size = path.stat().st_size
        except OSError:
            return False
        if size > self.max_file_bytes:
            return False
        if self._is_binary(path):
            return False
        return True

    def try_index(self, path: Path) -> bool:
        with self._lock:
            if not self._should_index(path):
                self.remove(path)
                return False

            stat = path.stat()
            rel_path = self._to_rel(path)
            digest = sha256_file(path)
            indexed_at = datetime.now(timezone.utc).isoformat()
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO files(path, mtime, size, file_type, sha256, indexed_at)
                    VALUES(?, ?, ?, ?, ?, ?)
                    ON CONFLICT(path) DO UPDATE SET
                      mtime = excluded.mtime,
                      size = excluded.size,
                      file_type = excluded.file_type,
                      sha256 = excluded.sha256,
                      indexed_at = excluded.indexed_at
                    """,
                    (
                        rel_path,
                        stat.st_mtime,
                        stat.st_size,
                        self._file_type(path),
                        digest,
                        indexed_at,
                    ),
                )
            return True

    def remove(self, path: Path) -> None:
        with self._lock:
            if not self._is_within_roots(path):
                return
            try:
                rel_path = self._to_rel(path)
            except ValueError:
                return
            with self._connect() as conn:
                conn.execute("DELETE FROM files WHERE path = ?", (rel_path,))

    def scan_all(self) -> tuple[int, int]:
        indexed = 0
        ignored = 0
        ignored += self._purge_stale_rows()
        for root in self.roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.is_dir():
                    continue
                if self.ignore_matcher.is_ignored(path):
                    ignored += 1
                    continue
                try:
                    if path.stat().st_size > self.max_file_bytes:
                        ignored += 1
                        continue
                except OSError:
                    ignored += 1
                    continue
                if self._is_binary(path):
                    ignored += 1
                    continue
                if self.try_index(path):
                    indexed += 1
                else:
                    ignored += 1

        self._set_meta("last_scan", datetime.now(timezone.utc).isoformat())
        self._set_meta("files_ignored", str(ignored))
        return indexed, ignored

    def _purge_stale_rows(self) -> int:
        with self._connect() as conn:
            rows = conn.execute("SELECT path FROM files").fetchall()

        stale: list[str] = []
        for row in rows:
            rel = str(row["path"])
            candidate = (self.workspace_root / rel).resolve()
            if not candidate.exists():
                stale.append(rel)
                continue
            if not self._is_within_roots(candidate):
                stale.append(rel)
                continue
            if self.ignore_matcher.is_ignored(candidate):
                stale.append(rel)
                continue
            try:
                if candidate.stat().st_size > self.max_file_bytes:
                    stale.append(rel)
                    continue
            except OSError:
                stale.append(rel)
                continue
            if self._is_binary(candidate):
                stale.append(rel)

        if not stale:
            return 0

        with self._connect() as conn:
            conn.executemany("DELETE FROM files WHERE path = ?", [(path,) for path in stale])
        return len(stale)

    def iter_indexed(
        self,
        scope_paths: list[str] | None = None,
        include_tests: bool = True,
    ) -> list[FileRecord]:
        clauses: list[str] = []
        params: list[str] = []

        if scope_paths:
            scope_clauses = []
            for scope in scope_paths:
                scope = scope.strip("/")
                if not scope:
                    continue
                scope_clauses.append("path LIKE ?")
                params.append(f"{scope}%")
            if scope_clauses:
                clauses.append("(" + " OR ".join(scope_clauses) + ")")

        if not include_tests:
            clauses.append("path NOT LIKE '%test%' AND path NOT LIKE 'tests/%'")

        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        query = f"SELECT path, mtime, size, file_type FROM files{where} ORDER BY path"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        records = [
            FileRecord(
                path=row["path"],
                mtime=float(row["mtime"]),
                size=int(row["size"]),
                file_type=row["file_type"],
            )
            for row in rows
        ]
        # Guard against stale indexed rows that are now ignored by current policy.
        return [
            record
            for record in records
            if not self.ignore_matcher.is_ignored(self.workspace_root / record.path)
        ]

    def get_record(self, rel_path: str) -> FileRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT path, mtime, size, file_type FROM files WHERE path = ?",
                (rel_path,),
            ).fetchone()
        if not row:
            return None
        return FileRecord(
            path=row["path"],
            mtime=float(row["mtime"]),
            size=int(row["size"]),
            file_type=row["file_type"],
        )

    def stats(self) -> dict:
        with self._connect() as conn:
            indexed = conn.execute("SELECT COUNT(*) AS n FROM files").fetchone()["n"]

        roots: list[str] = []
        for root in self.roots:
            rel = root.relative_to(self.workspace_root).as_posix()
            roots.append("./" if rel == "." else rel)

        return {
            "roots": roots,
            "last_scan": self._get_meta("last_scan", ""),
            "files_indexed": int(indexed),
            "files_ignored": int(self._get_meta("files_ignored", "0") or "0"),
            "ignores": self.ignore_matcher.effective_ignores(),
        }

    def start_watching(self) -> None:
        if self._observer is not None:
            return
        if Observer is None and PollingObserver is None:
            return

        handler = _IndexUpdateHandler(self)
        observer_cls = PollingObserver or Observer
        polling = observer_cls()
        try:
            for root in self.roots:
                if root.exists():
                    polling.schedule(handler, str(root), recursive=True)
            polling.start()
            self._observer = polling
        except Exception as exc:
            # Never fail MCP startup because filesystem watching is unavailable on a path.
            LOG.error("Filesystem watcher disabled due to startup error: %s", exc)
            try:
                polling.stop()
                polling.join(timeout=1)
            except Exception:
                pass
            self._observer = None

    def stop_watching(self) -> None:
        if self._observer is None:
            return
        self._observer.stop()
        self._observer.join(timeout=2)
        self._observer = None

    def abs_path(self, rel_path: str) -> Path:
        path = (self.workspace_root / rel_path).resolve()
        if not self._is_within_roots(path):
            raise ValueError("path is outside configured roots")
        return path

    def existing_paths(self) -> Iterable[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT path FROM files ORDER BY path").fetchall()
        for row in rows:
            rel = str(row["path"])
            if self.ignore_matcher.is_ignored(self.workspace_root / rel):
                continue
            yield rel

    def record_query_usage(self, *, tool: str, query: str) -> None:
        cleaned = " ".join(str(query or "").strip().split())
        if not cleaned:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO query_usage(tool, query, count, last_used)
                VALUES(?, ?, 1, ?)
                ON CONFLICT(tool, query) DO UPDATE SET
                    count = query_usage.count + 1,
                    last_used = excluded.last_used
                """,
                (tool, cleaned[:300], now),
            )

    def top_queries(self, *, limit: int = 12) -> list[dict[str, object]]:
        capped = max(1, min(limit, 50))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT tool, query, count, last_used
                FROM query_usage
                ORDER BY count DESC, last_used DESC
                LIMIT ?
                """,
                (capped,),
            ).fetchall()
        return [
            {
                "tool": str(row["tool"]),
                "query": str(row["query"]),
                "count": int(row["count"]),
                "last_used": str(row["last_used"]),
            }
            for row in rows
        ]
