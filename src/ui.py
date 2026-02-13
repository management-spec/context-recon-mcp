from __future__ import annotations

import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

_UI_INDEX_PATH = Path(__file__).resolve().parents[1] / "ui" / "index.html"

def start_dashboard(
    host: str,
    port: int,
    status_provider: Callable[..., dict],
    events_provider: Callable[..., dict] | None = None,
    event_waiter: Callable[[int, float], dict] | None = None,
    event_result_provider: Callable[[int], dict] | None = None,
    explorer_provider: Callable[[], dict] | None = None,
    actions: dict[str, Callable[[dict[str, Any]], dict]] | None = None,
) -> ThreadingHTTPServer:
    class DashboardHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send_json(self, payload: dict, status_code: int = 200) -> None:
            raw = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.end_headers()
            self.wfile.write(raw)

        def _send_html(self, raw: bytes, status_code: int = 200) -> None:
            self.send_response(status_code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.end_headers()
            self.wfile.write(raw)

        def _dashboard_html(self) -> tuple[bytes, int]:
            if not _UI_INDEX_PATH.exists():
                error = (
                    "Dashboard HTML missing. Expected file: "
                    f"{_UI_INDEX_PATH.as_posix()}"
                )
                return error.encode("utf-8"), 500
            raw = _UI_INDEX_PATH.read_bytes()
            return raw, 200

        def _ui_manifest(self) -> dict[str, Any]:
            if not _UI_INDEX_PATH.exists():
                return {
                    "exists": False,
                    "ui_path": _UI_INDEX_PATH.as_posix(),
                }
            raw = _UI_INDEX_PATH.read_bytes()
            stat = _UI_INDEX_PATH.stat()
            return {
                "exists": True,
                "ui_path": _UI_INDEX_PATH.as_posix(),
                "size_bytes": int(stat.st_size),
                "mtime_epoch": float(stat.st_mtime),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }

        def _read_json_body(self) -> dict[str, Any]:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                return {}
            raw = self.rfile.read(content_length)
            if not raw:
                return {}
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            return payload

        def _execute_action(self, key: str, payload: dict[str, Any]) -> None:
            if not actions or key not in actions:
                self._send_json({"error": "not_found"}, status_code=404)
                return
            try:
                result = actions[key](payload)
            except ValueError as exc:
                self._send_json({"error": "bad_request", "message": str(exc)}, status_code=400)
                return
            except FileNotFoundError as exc:
                self._send_json({"error": "not_found", "message": str(exc)}, status_code=404)
                return
            except Exception as exc:
                self._send_json({"error": "internal_error", "message": str(exc)}, status_code=500)
                return
            self._send_json(result)

        def _events_payload(self, *, limit: int, since_id: int) -> dict[str, Any]:
            if events_provider is None:
                return {"events": [], "last_id": since_id}
            try:
                return events_provider(limit=limit, since_id=since_id)
            except TypeError:
                payload = events_provider(limit)
                if isinstance(payload, dict):
                    payload.setdefault("last_id", since_id)
                    return payload
                return {"events": [], "last_id": since_id}

        def _send_event_stream(self, since_id: int) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            last_id = since_id
            try:
                while True:
                    if event_waiter is not None:
                        payload = event_waiter(last_id, 20.0)
                    else:
                        time.sleep(1.0)
                        payload = self._events_payload(limit=500, since_id=last_id)

                    events = list(payload.get("events", []))
                    if not events:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        continue

                    raw = json.dumps(payload, ensure_ascii=True).encode("utf-8")
                    self.wfile.write(b"data: " + raw + b"\n\n")
                    self.wfile.flush()
                    last_id = int(payload.get("last_id", last_id))
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                html, status_code = self._dashboard_html()
                self._send_html(html, status_code=status_code)
                return
            if parsed.path == "/api/status":
                params = parse_qs(parsed.query)
                force = params.get("force", ["0"])[0] in {"1", "true", "yes"}
                try:
                    payload = status_provider(force=force)
                except TypeError:
                    payload = status_provider()
                self._send_json(payload)
                return
            if parsed.path == "/api/events":
                params = parse_qs(parsed.query)
                try:
                    limit = int(params.get("limit", ["80"])[0])
                except ValueError:
                    limit = 80
                try:
                    since_id = int(params.get("since", ["0"])[0])
                except ValueError:
                    since_id = 0
                self._send_json(self._events_payload(limit=limit, since_id=since_id))
                return
            if parsed.path == "/api/event_result":
                if event_result_provider is None:
                    self._send_json({"error": "not_found"}, status_code=404)
                    return
                params = parse_qs(parsed.query)
                try:
                    event_id = int(params.get("id", ["0"])[0])
                except ValueError:
                    event_id = 0
                if event_id <= 0:
                    self._send_json({"error": "bad_request", "message": "id must be a positive integer"}, status_code=400)
                    return
                self._send_json(event_result_provider(event_id))
                return
            if parsed.path == "/api/events/stream":
                params = parse_qs(parsed.query)
                try:
                    since_id = int(params.get("since", ["0"])[0])
                except ValueError:
                    since_id = 0
                self._send_event_stream(since_id)
                return
            if parsed.path == "/api/explorer":
                if explorer_provider is None:
                    self._send_json({})
                    return
                self._send_json(explorer_provider())
                return
            if parsed.path == "/api/ui_manifest":
                self._send_json(self._ui_manifest())
                return
            self._send_json({"error": "not_found"}, status_code=404)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                payload = self._read_json_body()
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                self._send_json({"error": "bad_request", "message": str(exc)}, status_code=400)
                return

            if parsed.path == "/api/context_recon":
                self._execute_action("context_recon", payload)
                return
            if parsed.path == "/api/code_search":
                self._execute_action("code_search", payload)
                return
            if parsed.path == "/api/file_slice":
                self._execute_action("file_slice", payload)
                return
            if parsed.path == "/api/reindex":
                self._execute_action("reindex", payload)
                return
            if parsed.path == "/api/tool_update":
                self._execute_action("tool_update", payload)
                return
            if parsed.path == "/api/mute_path":
                self._execute_action("mute_path", payload)
                return

            self._send_json({"error": "not_found"}, status_code=404)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer((host, port), DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
