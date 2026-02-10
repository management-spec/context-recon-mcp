from __future__ import annotations

import atexit
from collections import Counter, deque
import hashlib
import logging
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from fastmcp import FastMCP

from gemini import GeminiConfig, GeminiReranker
from indexer import ContextIndexer
from search import CodeSearcher
from tools import code_search, context_recon, file_slice, index_inspection, project_overview, relevant_code
from ui import start_dashboard


LOG = logging.getLogger(__name__)


MCP_SERVER_COMMAND_MARKERS = (
    "/context-recon-mcp/src/server.py",
    "/context_recon_mcp/src/server.py",
    "/gemini-context-engine-mcp/src/server.py",
    "/gemini_context_engine_mcp/src/server.py",
)


def _is_managed_mcp_command(command: str) -> bool:
    return any(marker in command for marker in MCP_SERVER_COMMAND_MARKERS)


def _list_managed_mcp_processes() -> list[dict[str, Any]]:
    try:
        proc = subprocess.run(
            ["ps", "-axo", "pid,ppid,command"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []
    if proc.returncode != 0:
        return []

    matches: list[dict[str, Any]] = []
    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("PID"):
            continue
        parts = line.split(maxsplit=2)
        if len(parts) != 3:
            continue
        pid_raw, ppid_raw, command = parts
        try:
            pid = int(pid_raw)
            ppid = int(ppid_raw)
        except ValueError:
            continue
        if not _is_managed_mcp_command(command):
            continue
        matches.append({"pid": pid, "ppid": ppid, "command": command})
    return matches


def cleanup_managed_mcp_servers(*, current_pid: int, orphan_only: bool = False) -> dict[str, Any]:
    candidates = _list_managed_mcp_processes()
    killed: list[int] = []
    failed: list[dict[str, Any]] = []
    skipped_current = False

    for item in candidates:
        pid = int(item.get("pid", 0))
        ppid = int(item.get("ppid", 0))
        if pid == current_pid:
            skipped_current = True
            continue
        if orphan_only and ppid != 1:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            killed.append(pid)
        except OSError as exc:
            failed.append({"pid": pid, "error": str(exc)})

    return {
        "matched": len(candidates),
        "attempted": len(killed) + len(failed),
        "killed": len(killed),
        "killed_pids": killed,
        "failed": failed,
        "orphan_only": orphan_only,
        "current_pid": current_pid,
        "skipped_current": skipped_current,
    }


def cleanup_orphaned_mcp_servers(current_pid: int) -> int:
    """Best-effort cleanup for orphaned local MCP server processes from prior sessions."""
    result = cleanup_managed_mcp_servers(current_pid=current_pid, orphan_only=True)
    return int(result.get("killed", 0))


class ContextEngine:
    def __init__(self, workspace_root: Path, config: dict[str, Any], server_home: Path | None = None) -> None:
        self.workspace_root = workspace_root.resolve()
        self.server_home = (server_home or self.workspace_root).resolve()
        self.config = config

        roots = config.get("roots", ["."])
        self.max_file_bytes = int(config.get("max_file_bytes", 5 * 1024 * 1024))
        self.max_excerpt_lines = int(config.get("max_excerpt_lines", 200))
        self.gemini_timeout_seconds = int(config.get("gemini_timeout_seconds", 30))
        gemini_max_output_bytes = int(config.get("gemini_max_output_bytes", 2 * 1024 * 1024))

        self._index_db_fallback_reason = ""
        self._index_db_primary_path = str(config.get("database_path", ".context_engine/index.db"))
        self.indexer = self._build_indexer(
            roots=roots,
            db_path=self._index_db_primary_path,
            ignore_patterns=config.get("ignore", []),
        )
        self.searcher = CodeSearcher(self.indexer)
        self.reranker = GeminiReranker(
            GeminiConfig(
                command=config.get("gemini_command", "gemini"),
                args=list(config.get("gemini_args", [])),
                timeout_seconds=self.gemini_timeout_seconds,
                max_output_bytes=gemini_max_output_bytes,
                quota_poll_seconds=int(config.get("gemini_quota_poll_seconds", 1800)),
                auto_monitor_poll=bool(config.get("gemini_auto_monitor_poll", False)),
                planning_max_catalog_paths=int(config.get("gemini_planning_max_catalog_paths", 48)),
                planning_cache_seconds=int(config.get("gemini_planning_cache_seconds", 600)),
                planning_strategy=str(config.get("gemini_planning_strategy", "embedded")),
                rerank_max_candidates=int(config.get("gemini_rerank_max_candidates", 8)),
                rerank_excerpt_chars=int(config.get("gemini_rerank_excerpt_chars", 280)),
                rerank_prompt_char_budget=int(config.get("gemini_rerank_prompt_char_budget", 9000)),
            )
        )
        ide_context_config = config.get("ide_context_sync", {})
        if not isinstance(ide_context_config, dict):
            ide_context_config = {}
        ide_env_override = os.environ.get("CONTEXT_RECON_IDE_SYNC", "").strip().lower()
        if ide_env_override in {"1", "true", "yes", "on"}:
            self.ide_context_sync_enabled = True
        elif ide_env_override in {"0", "false", "no", "off"}:
            self.ide_context_sync_enabled = False
        else:
            self.ide_context_sync_enabled = bool(ide_context_config.get("enabled", True))
        ui_config = config.get("ui", {})
        if not isinstance(ui_config, dict):
            ui_config = {}
        self.ui_enabled = bool(ui_config.get("enabled", True))
        self.ui_host = str(ui_config.get("host", "127.0.0.1"))
        self.ui_port = int(ui_config.get("port", 8765))
        self.ui_auto_open = bool(ui_config.get("auto_open", True))
        self._ui_server: Any | None = None
        self._ui_opened = False
        self._activity_lock = threading.Lock()
        self._activity_condition = threading.Condition(self._activity_lock)
        self._activity: deque[dict[str, Any]] = deque(maxlen=500)
        self._activity_seq = 0
        self._result_snapshots: dict[int, dict[str, Any]] = {}
        self._snapshot_order: deque[int] = deque(maxlen=250)
        self._last_query: str = ""
        self._muted_prefixes: set[str] = set()
        self._hot_paths: dict[str, float] = {}
        self._recent_queries: deque[dict[str, Any]] = deque(maxlen=50)
        self._cooccurrence: Counter[tuple[str, str]] = Counter()
        self._active_context_tokens = 0
        self._context_window_tokens = int(ui_config.get("context_window_tokens", 128000))
        orphan_default = bool(config.get("orphan_shutdown_enabled", False))
        orphan_env = os.environ.get("CONTEXT_RECON_ORPHAN_SHUTDOWN", "").strip().lower()
        if orphan_env in {"1", "true", "yes", "on"}:
            self._orphan_shutdown_enabled = True
        elif orphan_env in {"0", "false", "no", "off"}:
            self._orphan_shutdown_enabled = False
        else:
            self._orphan_shutdown_enabled = orphan_default
        self._idle_shutdown_seconds = max(0, int(config.get("idle_shutdown_seconds", 900)))
        self._last_activity_monotonic = time.monotonic()
        self._activity_touch_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._stop_lock = threading.Lock()
        self._stopped = False
        self._initial_scan_thread: threading.Thread | None = None
        self._reindex_lock = threading.Lock()
        self._reindex_active = False
        self._reindex_started_at = ""
        self._reindex_finished_at = ""
        self._reindex_last_result: dict[str, Any] = {}
        self._reindex_last_error = ""

    def _build_indexer(
        self,
        *,
        roots: list[str],
        db_path: str,
        ignore_patterns: list[str],
    ) -> ContextIndexer:
        try:
            primary_indexer = ContextIndexer(
                workspace_root=self.workspace_root,
                roots=roots,
                db_path=db_path,
                max_file_bytes=self.max_file_bytes,
                ignore_patterns=ignore_patterns,
            )
            primary_indexer.assert_writable()
            return primary_indexer
        except (sqlite3.Error, OSError, PermissionError) as exc:
            workspace_key = hashlib.sha256(str(self.workspace_root).encode("utf-8")).hexdigest()[:12]
            fallback_db_path = (self.server_home / ".context_engine" / f"index-{workspace_key}.db").resolve()
            fallback_indexer = ContextIndexer(
                workspace_root=self.workspace_root,
                roots=roots,
                db_path=str(fallback_db_path),
                max_file_bytes=self.max_file_bytes,
                ignore_patterns=ignore_patterns,
            )
            fallback_indexer.assert_writable()
            self._index_db_fallback_reason = (
                f"Primary index database was not writable ({type(exc).__name__}: {exc}). "
                f"Using fallback database at {fallback_db_path}."
            )
            LOG.error(self._index_db_fallback_reason)
            return fallback_indexer

    def start(self) -> None:
        self._touch_activity()
        self._start_initial_scan()
        threading.Thread(target=self.indexer.start_watching, daemon=True).start()
        if self._orphan_shutdown_enabled:
            cleaned = cleanup_orphaned_mcp_servers(os.getpid())
            if cleaned:
                LOG.info("Cleaned %s orphaned MCP server process(es).", cleaned)
        threading.Thread(target=self._lifecycle_watchdog, daemon=True).start()
        if self.ui_enabled and self._ui_server is None:
            try:
                self._ui_server = start_dashboard(
                    host=self.ui_host,
                    port=self.ui_port,
                    status_provider=self.dashboard_status,
                    events_provider=self.recent_activity,
                    event_waiter=self.wait_for_activity,
                    event_result_provider=self.get_event_result,
                    explorer_provider=self.explorer_snapshot,
                    actions={
                        "context_recon": self.ui_context_recon,
                        "code_search": self.ui_code_search,
                        "file_slice": self.ui_file_slice,
                        "reindex": self.ui_reindex,
                        "tool_cleanup": self.ui_tool_cleanup,
                        "mute_path": self.ui_mute_path,
                    },
                )
                LOG.info("Context_Recon_MCP dashboard running at http://%s:%s", self.ui_host, self.ui_port)
                if self.ui_auto_open and not self._ui_opened:
                    self._ui_opened = True
                    dashboard_url = f"http://{self.ui_host}:{self.ui_port}"
                    threading.Thread(
                        target=self._open_dashboard,
                        args=(dashboard_url,),
                        daemon=True,
                    ).start()
            except OSError as exc:
                LOG.warning("Dashboard did not start on %s:%s (%s)", self.ui_host, self.ui_port, exc)

    def stop(self) -> None:
        with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True
            self._shutdown_event.set()
        self.indexer.stop_watching()
        if self._ui_server is not None:
            self._ui_server.shutdown()
            self._ui_server.server_close()
            self._ui_server = None

    def _start_initial_scan(self) -> None:
        if self._initial_scan_thread is not None:
            return
        self._initial_scan_thread = threading.Thread(target=self._run_initial_scan, daemon=True)
        self._initial_scan_thread.start()

    def _run_initial_scan(self) -> None:
        try:
            indexed, ignored = self.indexer.scan_all()
            LOG.info("Initial index scan completed (%s indexed, %s ignored).", indexed, ignored)
        except Exception:
            LOG.exception("Initial index scan failed.")

    def _touch_activity(self) -> None:
        with self._activity_touch_lock:
            self._last_activity_monotonic = time.monotonic()

    def _shutdown_process(self, reason: str) -> None:
        LOG.info("Shutting down Context_Recon_MCP process: %s", reason)
        self.stop()
        os._exit(0)

    def _lifecycle_watchdog(self) -> None:
        check_interval_seconds = 15.0
        while not self._shutdown_event.wait(timeout=check_interval_seconds):
            ppid = os.getppid()
            now = time.monotonic()
            with self._activity_touch_lock:
                idle_for = now - self._last_activity_monotonic
            if self._orphan_shutdown_enabled and ppid == 1:
                self._shutdown_process("parent exited (orphan process)")
            if self._idle_shutdown_seconds > 0 and idle_for >= self._idle_shutdown_seconds:
                self._shutdown_process(f"idle timeout reached ({int(idle_for)}s)")

    def dashboard_status(self, force: bool = False) -> dict[str, Any]:
        self._touch_activity()
        return {
            "server_name": "Context_Recon_MCP",
            "workspace_root": str(self.workspace_root),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "active_context": self._last_query,
            "session": {
                "active_context_tokens": self._active_context_tokens,
                "context_window_tokens": self._context_window_tokens,
                "muted_prefixes": sorted(self._muted_prefixes),
            },
            "index": self.indexer.stats(),
            "index_db": {
                "path": str(self.indexer.db_path),
                "fallback_reason": self._index_db_fallback_reason,
            },
            "ide_context_sync": {
                "enabled": self.ide_context_sync_enabled,
            },
            "reindex": self._reindex_status(),
            "gemini": self.reranker.status(force=force),
        }

    def _reindex_status(self) -> dict[str, Any]:
        with self._reindex_lock:
            return {
                "active": self._reindex_active,
                "started_at": self._reindex_started_at,
                "finished_at": self._reindex_finished_at,
                "last_error": self._reindex_last_error,
                "last_result": dict(self._reindex_last_result),
            }

    def _record_activity(
        self,
        *,
        tool: str,
        source: str,
        status: str,
        summary: str,
        meta: dict[str, Any] | None = None,
    ) -> int:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": tool,
            "source": source,
            "status": status,
            "summary": summary,
            "meta": meta or {},
        }
        with self._activity_condition:
            self._activity_seq += 1
            event["id"] = self._activity_seq
            self._activity.append(event)
            self._activity_condition.notify_all()
            return int(event["id"])

    def _store_result_snapshot(self, *, event_id: int, payload: dict[str, Any]) -> None:
        with self._activity_lock:
            if self._snapshot_order.maxlen and len(self._snapshot_order) >= self._snapshot_order.maxlen:
                oldest_id = self._snapshot_order[0]
                self._result_snapshots.pop(oldest_id, None)
            self._result_snapshots[event_id] = payload
            self._snapshot_order.append(event_id)

    def get_event_result(self, event_id: int) -> dict[str, Any]:
        with self._activity_lock:
            payload = self._result_snapshots.get(event_id)
        if payload is None:
            return {"event_id": event_id, "available": False}
        return {"event_id": event_id, "available": True, **payload}

    def recent_activity(self, limit: int = 80, since_id: int = 0) -> dict[str, Any]:
        bounded = max(1, min(limit, 500))
        with self._activity_lock:
            if since_id > 0:
                events = [event for event in self._activity if int(event.get("id", 0)) > since_id]
            else:
                events = list(self._activity)[-bounded:]
            last_id = int(events[-1]["id"]) if events else self._activity_seq
        return {"events": events, "last_id": last_id}

    def wait_for_activity(self, since_id: int, timeout_seconds: float = 20.0) -> dict[str, Any]:
        deadline = max(0.1, min(timeout_seconds, 60.0))
        with self._activity_condition:
            if self._activity_seq <= since_id:
                self._activity_condition.wait(timeout=deadline)
        return self.recent_activity(limit=500, since_id=since_id)

    def _normalize_prefix(self, raw_prefix: str) -> str:
        prefix = raw_prefix.strip().strip("/")
        return prefix

    def _normalize_scope_paths(self, scope_paths: str | list[str] | None) -> list[str] | None:
        if scope_paths is None:
            return None

        if isinstance(scope_paths, str):
            normalized = [part.strip() for part in scope_paths.split(",") if part.strip()]
            return normalized or None

        if isinstance(scope_paths, list):
            normalized: list[str] = []
            for part in scope_paths:
                if not isinstance(part, str):
                    continue
                cleaned = part.strip()
                if cleaned:
                    normalized.append(cleaned)
            return normalized or None

        return None

    def _is_muted_path(self, path: str) -> bool:
        for prefix in self._muted_prefixes:
            if path == prefix or path.startswith(prefix + "/"):
                return True
        return False

    def _apply_muted_scope(self, scope_paths: list[str] | None) -> list[str] | None:
        if not self._muted_prefixes:
            return scope_paths
        if not scope_paths:
            scope_paths = [root.relative_to(self.workspace_root).as_posix() for root in self.indexer.roots]

        effective: list[str] = []
        for scope in scope_paths:
            scope_norm = self._normalize_prefix(scope)
            if not scope_norm:
                effective.append(scope)
                continue
            if self._is_muted_path(scope_norm):
                continue
            effective.append(scope)
        return effective or scope_paths

    def _touch_paths(self, paths: list[str]) -> None:
        if not paths:
            return
        now = time.time()
        for path in paths:
            self._hot_paths[path] = now
        cutoff = now - 300
        stale = [path for path, seen_at in self._hot_paths.items() if seen_at < cutoff]
        for path in stale:
            self._hot_paths.pop(path, None)

    def _estimate_snippet_tokens(self, snippets: list[dict[str, Any]]) -> int:
        total_chars = sum(len(str(item.get("excerpt", ""))) for item in snippets)
        return max(0, total_chars // 4)

    def _record_recon_intel(self, *, query: str, result: dict[str, Any]) -> None:
        snippets = list(result.get("snippets", []))
        if not snippets:
            self._active_context_tokens = 0
            return
        token_estimate = self._estimate_snippet_tokens(snippets)
        self._active_context_tokens = token_estimate
        path_scores: dict[str, float] = {}
        for item in snippets:
            path = str(item.get("path", ""))
            if not path:
                continue
            path_scores[path] = max(path_scores.get(path, 0.0), float(item.get("score", 0.0)))
        self._recent_queries.append(
            {
                "query": query,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "path_scores": path_scores,
                "coverage": result.get("coverage", {}),
                "token_estimate": token_estimate,
            }
        )
        paths = sorted(path_scores.keys())
        self._touch_paths(paths)
        for idx, left in enumerate(paths):
            for right in paths[idx + 1 :]:
                self._cooccurrence[(left, right)] += 1

    def run_code_search(
        self,
        *,
        pattern: str,
        regex: bool,
        case_sensitive: bool,
        scope_paths: list[str] | None,
        max_hits: int,
        source: str,
    ) -> dict:
        self._touch_activity()
        started = time.perf_counter()
        effective_scope = scope_paths
        if source == "dashboard":
            effective_scope = self._apply_muted_scope(scope_paths)
        try:
            result = code_search.run(
                searcher=self.searcher,
                indexer=self.indexer,
                reranker=self.reranker,
                pattern=pattern,
                regex=regex,
                case_sensitive=case_sensitive,
                scope_paths=effective_scope,
                max_hits=max_hits,
            )
            if source == "dashboard" and self._muted_prefixes:
                result["hits"] = [hit for hit in result.get("hits", []) if not self._is_muted_path(str(hit.get("path", "")))]
            self.indexer.record_query_usage(tool="context.code_search", query=pattern)
            self._touch_paths([str(hit.get("path", "")) for hit in result.get("hits", [])[:30]])
            duration_ms = int((time.perf_counter() - started) * 1000)
            event_id = self._record_activity(
                tool="context.code_search",
                source=source,
                status="ok",
                summary=f"{len(result.get('hits', []))} hits for '{pattern}'",
                meta={"duration_ms": duration_ms, "scope_paths": effective_scope or []},
            )
            self._store_result_snapshot(
                event_id=event_id,
                payload={
                    "tool": "context.code_search",
                    "query": pattern,
                    "hits": list(result.get("hits", []))[:50],
                    "retrieval_plan": result.get("retrieval_plan", {}),
                },
            )
            return result
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            self._record_activity(
                tool="context.code_search",
                source=source,
                status="error",
                summary=f"code_search failed: {exc}",
                meta={"duration_ms": duration_ms},
            )
            raise

    def run_relevant_code(
        self,
        *,
        query: str,
        scope_paths: list[str] | None,
        max_results: int,
        max_excerpt_lines: int,
        include_tests: bool,
        source: str,
    ) -> dict:
        self._touch_activity()
        started = time.perf_counter()
        bounded_excerpt_lines = min(max_excerpt_lines, self.max_excerpt_lines)
        self._last_query = query
        effective_scope = scope_paths
        if source == "dashboard":
            effective_scope = self._apply_muted_scope(scope_paths)
        try:
            result = relevant_code.run(
                indexer=self.indexer,
                searcher=self.searcher,
                reranker=self.reranker,
                query=query,
                scope_paths=effective_scope,
                max_results=max_results,
                max_excerpt_lines=bounded_excerpt_lines,
                include_tests=include_tests,
            )
            if source == "dashboard" and self._muted_prefixes:
                filtered = [snippet for snippet in result.get("snippets", []) if not self._is_muted_path(str(snippet.get("path", "")))]
                result["snippets"] = filtered[:max_results]
            self.indexer.record_query_usage(tool="context.relevant_code", query=query)
            self._record_recon_intel(query=query, result=result)
            duration_ms = int((time.perf_counter() - started) * 1000)
            event_id = self._record_activity(
                tool="context.relevant_code",
                source=source,
                status="ok",
                summary=f"{len(result.get('snippets', []))} snippets for '{query}'",
                meta={"duration_ms": duration_ms, "scope_paths": effective_scope or []},
            )
            self._store_result_snapshot(
                event_id=event_id,
                payload={
                    "tool": "context.relevant_code",
                    "query": query,
                    "snippets": list(result.get("snippets", []))[:max_results],
                    "coverage": result.get("coverage", {}),
                    "retrieval_plan": result.get("retrieval_plan", {}),
                },
            )
            return result
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            self._record_activity(
                tool="context.relevant_code",
                source=source,
                status="error",
                summary=f"relevant_code failed: {exc}",
                meta={"duration_ms": duration_ms},
            )
            raise

    def run_context_recon(
        self,
        *,
        query: str | None,
        question: str | None,
        scope_paths: list[str] | None,
        max_results: int,
        max_excerpt_lines: int,
        include_tests: bool,
        source: str,
    ) -> dict:
        self._touch_activity()
        started = time.perf_counter()
        resolved_query = (query or question or "").strip()
        if resolved_query:
            self._last_query = resolved_query
        bounded_excerpt_lines = min(max_excerpt_lines, self.max_excerpt_lines)
        effective_scope = scope_paths
        if source == "dashboard":
            effective_scope = self._apply_muted_scope(scope_paths)
        try:
            result = context_recon.run(
                indexer=self.indexer,
                searcher=self.searcher,
                reranker=self.reranker,
                query=query,
                question=question,
                scope_paths=effective_scope,
                max_results=max_results,
                max_excerpt_lines=bounded_excerpt_lines,
                include_tests=include_tests,
            )
            if source == "dashboard" and self._muted_prefixes:
                filtered = [snippet for snippet in result.get("snippets", []) if not self._is_muted_path(str(snippet.get("path", "")))]
                result["snippets"] = filtered[:max_results]
            if resolved_query:
                self.indexer.record_query_usage(tool="context.context_recon", query=resolved_query)
            self._record_recon_intel(query=resolved_query or "<empty>", result=result)
            duration_ms = int((time.perf_counter() - started) * 1000)
            summary_query = resolved_query or "<empty>"
            event_id = self._record_activity(
                tool="context.context_recon",
                source=source,
                status="ok",
                summary=f"{len(result.get('snippets', []))} snippets for '{summary_query}'",
                meta={"duration_ms": duration_ms, "scope_paths": effective_scope or []},
            )
            self._store_result_snapshot(
                event_id=event_id,
                payload={
                    "tool": "context.context_recon",
                    "query": summary_query,
                    "snippets": list(result.get("snippets", []))[:max_results],
                    "coverage": result.get("coverage", {}),
                    "retrieval_plan": result.get("retrieval_plan", {}),
                },
            )
            return result
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            self._record_activity(
                tool="context.context_recon",
                source=source,
                status="error",
                summary=f"context_recon failed: {exc}",
                meta={"duration_ms": duration_ms},
            )
            raise

    def run_file_slice(self, *, path: str, start_line: int, end_line: int, source: str) -> dict:
        self._touch_activity()
        started = time.perf_counter()
        try:
            result = file_slice.run(
                indexer=self.indexer,
                path=path,
                start_line=start_line,
                end_line=end_line,
                max_excerpt_lines=min(self.max_excerpt_lines, 200),
            )
            self._touch_paths([path])
            duration_ms = int((time.perf_counter() - started) * 1000)
            self._record_activity(
                tool="context.file_slice",
                source=source,
                status="ok",
                summary=f"{path}:{result.get('start_line')}:{result.get('end_line')}",
                meta={"duration_ms": duration_ms},
            )
            return result
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            self._record_activity(
                tool="context.file_slice",
                source=source,
                status="error",
                summary=f"file_slice failed: {exc}",
                meta={"duration_ms": duration_ms},
            )
            raise

    def run_reindex(self, *, source: str) -> dict:
        self._touch_activity()
        started = time.perf_counter()
        indexed, ignored = self.indexer.scan_all()
        duration_ms = int((time.perf_counter() - started) * 1000)
        self._record_activity(
            tool="context.reindex",
            source=source,
            status="ok",
            summary=f"Reindexed: {indexed} indexed, {ignored} ignored",
            meta={"duration_ms": duration_ms},
        )
        return {
            "indexed": indexed,
            "ignored": ignored,
            "stats": self.indexer.stats(),
        }

    def run_tool_cleanup(self, *, source: str, orphan_only: bool = False) -> dict:
        self._touch_activity()
        started = time.perf_counter()
        result = cleanup_managed_mcp_servers(current_pid=os.getpid(), orphan_only=orphan_only)
        duration_ms = int((time.perf_counter() - started) * 1000)
        self._record_activity(
            tool="context.tool_cleanup",
            source=source,
            status="ok",
            summary=f"Killed {result.get('killed', 0)} managed MCP process(es)",
            meta={"duration_ms": duration_ms, "orphan_only": orphan_only},
        )
        return result

    def _run_reindex_background(self, *, source: str) -> None:
        self._touch_activity()
        started = time.perf_counter()
        try:
            indexed, ignored = self.indexer.scan_all()
            duration_ms = int((time.perf_counter() - started) * 1000)
            result = {
                "indexed": indexed,
                "ignored": ignored,
                "stats": self.indexer.stats(),
                "duration_ms": duration_ms,
            }
            with self._reindex_lock:
                self._reindex_last_result = result
                self._reindex_last_error = ""
                self._reindex_finished_at = datetime.now(timezone.utc).isoformat()
            self._record_activity(
                tool="context.reindex",
                source=source,
                status="ok",
                summary=f"Reindexed: {indexed} indexed, {ignored} ignored",
                meta={"duration_ms": duration_ms},
            )
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            with self._reindex_lock:
                self._reindex_last_error = str(exc)
                self._reindex_finished_at = datetime.now(timezone.utc).isoformat()
            self._record_activity(
                tool="context.reindex",
                source=source,
                status="error",
                summary=f"reindex failed: {exc}",
                meta={"duration_ms": duration_ms},
            )
        finally:
            with self._reindex_lock:
                self._reindex_active = False

    def start_reindex(self, *, source: str) -> dict:
        self._touch_activity()
        with self._reindex_lock:
            if self._reindex_active:
                return {"started": False, "active": True, "message": "Re-index already running."}
            self._reindex_active = True
            self._reindex_started_at = datetime.now(timezone.utc).isoformat()
            self._reindex_finished_at = ""
            self._reindex_last_error = ""
            self._reindex_last_result = {}

        self._record_activity(
            tool="context.reindex",
            source=source,
            status="ok",
            summary="Re-index started",
        )
        threading.Thread(
            target=self._run_reindex_background,
            kwargs={"source": source},
            daemon=True,
        ).start()
        return {"started": True, "active": True, "message": "Re-index started."}

    def _heatmap_scores(self, recent_count: int = 5) -> dict[str, float]:
        weights: dict[str, float] = {}
        recent = list(self._recent_queries)[-recent_count:]
        for item in recent:
            for path, score in dict(item.get("path_scores", {})).items():
                weights[path] = weights.get(path, 0.0) + float(score)
        return weights

    def _topology_edges(self, limit: int = 16) -> list[dict[str, Any]]:
        edges = sorted(self._cooccurrence.items(), key=lambda pair: pair[1], reverse=True)[:limit]
        return [
            {"a": left, "b": right, "weight": int(weight)}
            for (left, right), weight in edges
        ]

    def _build_tree(self, paths: list[str], heatmap: dict[str, float]) -> list[dict[str, Any]]:
        now = time.time()
        root: dict[str, Any] = {"name": ".", "path": "", "kind": "dir", "children": {}}
        for rel_path in paths:
            node = root
            parts = [part for part in rel_path.split("/") if part]
            if any(part.startswith(".") for part in parts):
                continue
            current_parts: list[str] = []
            for idx, part in enumerate(parts):
                current_parts.append(part)
                child_path = "/".join(current_parts)
                is_file = idx == len(parts) - 1
                children = node.setdefault("children", {})
                if part not in children:
                    children[part] = {
                        "name": part,
                        "path": child_path,
                        "kind": "file" if is_file else "dir",
                        "children": {},
                    }
                node = children[part]

        def walk(node: dict[str, Any]) -> dict[str, Any]:
            path = str(node.get("path", ""))
            children_map = node.get("children", {})
            children = [walk(child) for _, child in sorted(children_map.items(), key=lambda item: item[0].lower())]
            is_file = node.get("kind") == "file"
            hot = False
            if is_file:
                hot = (now - self._hot_paths.get(path, 0.0)) <= 120
            else:
                hot = any(child.get("hot", False) for child in children)
            muted = self._is_muted_path(path) if path else False
            indexed_count = 1 if is_file else sum(int(child.get("indexed_count", 0)) for child in children)
            relevance = float(heatmap.get(path, 0.0))
            if not is_file and relevance <= 0:
                relevance = sum(float(child.get("relevance", 0.0)) for child in children)
            return {
                "name": node.get("name", ""),
                "path": path,
                "kind": node.get("kind", "dir"),
                "hot": hot,
                "muted": muted,
                "indexed_count": indexed_count,
                "relevance": relevance,
                "children": children,
            }

        transformed = walk(root)
        return transformed.get("children", [])

    def explorer_snapshot(self) -> dict[str, Any]:
        indexed_paths = list(self.indexer.existing_paths())
        heatmap = self._heatmap_scores(recent_count=5)
        index_stats = self.indexer.stats()
        indexed = int(index_stats.get("files_indexed", 0))
        ignored = int(index_stats.get("files_ignored", 0))
        total_known = max(1, indexed + ignored)
        known_pct = (indexed / total_known) * 100.0
        ignored_pct = (ignored / total_known) * 100.0
        unknown_pct = max(0.0, 100.0 - known_pct - ignored_pct)
        if self._recent_queries:
            latest = self._recent_queries[-1]
            coverage = latest.get("coverage", {})
            candidates = int(dict(coverage).get("candidates_scanned", 0) or 0)
            snippets = max(1, len(dict(latest).get("path_scores", {})))
            ratio = candidates / snippets
            if ratio >= 18:
                efficiency = {"score": 40, "label": "too_broad", "ratio": ratio}
            elif ratio <= 2:
                efficiency = {"score": 55, "label": "too_narrow", "ratio": ratio}
            else:
                efficiency = {"score": 84, "label": "balanced", "ratio": ratio}
        else:
            efficiency = {"score": 0, "label": "no_data", "ratio": 0.0}

        heatmap_rows = sorted(heatmap.items(), key=lambda item: item[1], reverse=True)[:20]
        return {
            "tree": self._build_tree(indexed_paths, heatmap),
            "heatmap": [{"path": path, "score": score} for path, score in heatmap_rows],
            "top_queries": self.indexer.top_queries(limit=10),
            "muted_prefixes": sorted(self._muted_prefixes),
            "coverage_graph": {
                "indexed_percent": round(known_pct, 2),
                "ignored_percent": round(ignored_pct, 2),
                "unknown_percent": round(unknown_pct, 2),
            },
            "topology_edges": self._topology_edges(limit=14),
            "token_efficiency": efficiency,
            "token_budget": {
                "active_context_tokens": self._active_context_tokens,
                "context_window_tokens": self._context_window_tokens,
            },
        }

    def ui_context_recon(self, payload: dict[str, Any]) -> dict:
        scope_paths = self._normalize_scope_paths(payload.get("scope_paths"))
        return self.run_context_recon(
            query=payload.get("query"),
            question=payload.get("question"),
            scope_paths=scope_paths,
            max_results=int(payload.get("max_results", 5)),
            max_excerpt_lines=int(payload.get("max_excerpt_lines", 150)),
            include_tests=bool(payload.get("include_tests", False)),
            source="dashboard",
        )

    def ui_code_search(self, payload: dict[str, Any]) -> dict:
        pattern = str(payload.get("pattern", "")).strip()
        if not pattern:
            raise ValueError("`pattern` is required")
        scope_paths = self._normalize_scope_paths(payload.get("scope_paths"))
        return self.run_code_search(
            pattern=pattern,
            regex=bool(payload.get("regex", False)),
            case_sensitive=bool(payload.get("case_sensitive", False)),
            scope_paths=scope_paths,
            max_hits=int(payload.get("max_hits", 50)),
            source="dashboard",
        )

    def ui_file_slice(self, payload: dict[str, Any]) -> dict:
        path = str(payload.get("path", "")).strip()
        if not path:
            raise ValueError("`path` is required")
        return self.run_file_slice(
            path=path,
            start_line=int(payload.get("start_line", 1)),
            end_line=int(payload.get("end_line", 200)),
            source="dashboard",
        )

    def ui_reindex(self, payload: dict[str, Any]) -> dict:
        _ = payload
        return self.start_reindex(source="dashboard")

    def ui_tool_cleanup(self, payload: dict[str, Any]) -> dict:
        orphan_only = bool(payload.get("orphan_only", False))
        return self.run_tool_cleanup(source="dashboard", orphan_only=orphan_only)

    def ui_mute_path(self, payload: dict[str, Any]) -> dict:
        self._touch_activity()
        raw_path = str(payload.get("path", "")).strip()
        if not raw_path:
            raise ValueError("`path` is required")
        normalized = self._normalize_prefix(raw_path)
        muted = bool(payload.get("muted", True))
        if muted:
            self._muted_prefixes.add(normalized)
            summary = f"Muted '{normalized}' for dashboard queries"
        else:
            self._muted_prefixes.discard(normalized)
            summary = f"Unmuted '{normalized}' for dashboard queries"
        self._record_activity(
            tool="context.mute_path",
            source="dashboard",
            status="ok",
            summary=summary,
        )
        return {"muted_prefixes": sorted(self._muted_prefixes)}

    @staticmethod
    def _open_dashboard(url: str) -> None:
        # First try the standard Python browser handler.
        try:
            opened = webbrowser.open(url, new=2)
            if opened:
                return
        except Exception as exc:  # pragma: no cover - platform/browser dependent
            LOG.debug("webbrowser.open failed for %s (%s)", url, exc)

        # Fallback to native launchers for environments where webbrowser.open returns false.
        launchers: list[list[str]] = []
        if sys.platform == "darwin":
            launchers.append(["open", url])
        elif sys.platform.startswith("linux"):
            launchers.append(["xdg-open", url])
        elif os.name == "nt":
            launchers.append(["cmd", "/c", "start", "", url])

        for command in launchers:
            try:
                subprocess.Popen(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                )
                return
            except Exception as exc:  # pragma: no cover - launcher availability differs per host
                LOG.debug("Dashboard launcher failed: %s (%s)", command[0], exc)

        LOG.warning(
            "Failed to auto-open dashboard URL %s. Open it manually in your browser.",
            url,
        )


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return raw or {}


def resolve_runtime_paths() -> tuple[Path, Path]:
    server_home = Path(
        os.environ.get("GEMINI_CONTEXT_SERVER_HOME", Path(__file__).resolve().parents[1].as_posix())
    ).resolve()
    configured_workspace = os.environ.get("GEMINI_CONTEXT_WORKSPACE_ROOT", "").strip()
    if configured_workspace:
        workspace_root = Path(configured_workspace).resolve()
    else:
        cwd = Path.cwd().resolve()
        # Desktop MCP hosts may spawn with "/" cwd; default to HOME in that case.
        workspace_root = cwd if cwd != Path("/") else Path.home().resolve()
    return server_home, workspace_root


def build_mcp(engine: ContextEngine) -> FastMCP:
    mcp = FastMCP("Context_Recon_MCP")

    @mcp.tool(name="context.project_overview")
    def tool_project_overview(
        scope_paths: str | list[str] | None = None,
        max_depth: int = 4,
    ) -> dict:
        result = project_overview.run(
            indexer=engine.indexer,
            scope_paths=engine._normalize_scope_paths(scope_paths),
            max_depth=max_depth,
        )
        engine._record_activity(
            tool="context.project_overview",
            source="mcp",
            status="ok",
            summary="Project overview generated",
        )
        return result

    @mcp.tool(name="context.index_inspection")
    def tool_index_inspection() -> dict:
        payload = index_inspection.run(
            indexer=engine.indexer,
            max_file_bytes=engine.max_file_bytes,
            max_excerpt_lines=engine.max_excerpt_lines,
            gemini_timeout_seconds=engine.gemini_timeout_seconds,
            gemini_status=engine.reranker.status(force=True),
        )
        payload["index_db"] = {
            "path": str(engine.indexer.db_path),
            "fallback_reason": engine._index_db_fallback_reason,
        }
        payload["ide_context_sync"] = {
            "enabled": engine.ide_context_sync_enabled,
        }
        payload["ui"] = {
            "enabled": engine.ui_enabled,
            "url": f"http://{engine.ui_host}:{engine.ui_port}" if engine.ui_enabled else "",
            "auto_open": engine.ui_auto_open,
        }
        engine._record_activity(
            tool="context.index_inspection",
            source="mcp",
            status="ok",
            summary="Index inspection requested",
        )
        return payload

    @mcp.tool(name="context.code_search")
    def tool_code_search(
        pattern: str,
        regex: bool = False,
        case_sensitive: bool = False,
        scope_paths: str | list[str] | None = None,
        max_hits: int = 50,
    ) -> dict:
        return engine.run_code_search(
            pattern=pattern,
            regex=regex,
            case_sensitive=case_sensitive,
            scope_paths=engine._normalize_scope_paths(scope_paths),
            max_hits=max_hits,
            source="mcp",
        )

    @mcp.tool(name="context.relevant_code")
    def tool_relevant_code(
        query: str,
        scope_paths: str | list[str] | None = None,
        max_results: int = 3,
        max_excerpt_lines: int = 80,
        include_tests: bool = False,
    ) -> dict:
        return engine.run_relevant_code(
            query=query,
            scope_paths=engine._normalize_scope_paths(scope_paths),
            max_results=max_results,
            max_excerpt_lines=max_excerpt_lines,
            include_tests=include_tests,
            source="mcp",
        )

    @mcp.tool(name="context.file_slice")
    def tool_file_slice(path: str, start_line: int, end_line: int) -> dict:
        return engine.run_file_slice(
            path=path,
            start_line=start_line,
            end_line=end_line,
            source="mcp",
        )

    @mcp.tool(name="context.context_recon")
    def tool_context_recon(
        query: str | None = None,
        question: str | None = None,
        scope_paths: str | list[str] | None = None,
        max_results: int = 3,
        max_excerpt_lines: int = 80,
        include_tests: bool = False,
    ) -> dict:
        return engine.run_context_recon(
            query=query,
            question=question,
            scope_paths=engine._normalize_scope_paths(scope_paths),
            max_results=max_results,
            max_excerpt_lines=max_excerpt_lines,
            include_tests=include_tests,
            source="mcp",
        )

    @mcp.tool(name="context.tool_cleanup")
    def tool_tool_cleanup(orphan_only: bool = False) -> dict:
        return engine.run_tool_cleanup(source="mcp", orphan_only=orphan_only)

    return mcp


def main() -> None:
    # Keep stdio transport quiet for MCP clients that are sensitive to noisy startup logs.
    logging.basicConfig(level=logging.ERROR)

    server_home, workspace_root = resolve_runtime_paths()
    config = load_config(server_home / "config.yaml")
    engine = ContextEngine(workspace_root, config, server_home=server_home)
    engine.start()
    atexit.register(engine.stop)

    mcp = build_mcp(engine)
    mcp.run(show_banner=False, log_level="ERROR")


if __name__ == "__main__":
    main()
