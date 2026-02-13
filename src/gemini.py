from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from utils.json_schema import validate_gemini_rerank_payload


class GeminiError(RuntimeError):
    pass


@dataclass(frozen=True)
class GeminiConfig:
    command: str
    args: list[str]
    timeout_seconds: int
    max_output_bytes: int
    quota_poll_seconds: int = 1800
    auto_monitor_poll: bool = False
    planning_max_catalog_paths: int = 48
    planning_cache_seconds: int = 600
    planning_strategy: str = "embedded"
    rerank_max_candidates: int = 8
    rerank_excerpt_chars: int = 280
    rerank_prompt_char_budget: int = 9000


class GeminiQuotaMonitor:
    _LOAD_CODE_ASSIST_URL = "https://cloudcode-pa.googleapis.com/v1internal:loadCodeAssist"
    _RETRIEVE_USER_QUOTA_URL = "https://cloudcode-pa.googleapis.com/v1internal:retrieveUserQuota"
    _TOKEN_REFRESH_URL = "https://oauth2.googleapis.com/token"

    def __init__(
        self,
        *,
        command: str,
        timeout_seconds: int,
        poll_interval_seconds: int,
    ) -> None:
        self.command = command
        self.timeout_seconds = max(3, min(timeout_seconds, 20))
        self.poll_interval_seconds = max(30, poll_interval_seconds)
        self._lock = threading.Lock()
        self._last_checked_monotonic = 0.0
        self._refresh_inflight = False
        self._state: dict[str, Any] = {
            "available": False,
            "checked": False,
            "checked_at": "",
            "auth_type": "unknown",
            "account_email": "",
            "tier": "",
            "project_id": "",
            "models": [],
            "lowest_percent_left": None,
            "last_error": "not_checked",
            "guidance": "Gemini quota monitor has not been checked yet.",
            "next_steps": [
                "Run context.index_inspection to trigger quota monitoring.",
            ],
        }

    def status(self, force: bool = False, allow_poll: bool = True) -> dict[str, Any]:
        if not allow_poll:
            with self._lock:
                return dict(self._state)
        if force:
            run_sync = False
            with self._lock:
                if not self._refresh_inflight:
                    self._refresh_inflight = True
                    run_sync = True
            if run_sync:
                self._refresh_once()
            else:
                deadline = time.monotonic() + float(self.timeout_seconds + 1)
                while time.monotonic() < deadline:
                    with self._lock:
                        if not self._refresh_inflight:
                            break
                    time.sleep(0.05)
            with self._lock:
                return dict(self._state)
        due = force
        now = time.monotonic()
        can_start = False
        with self._lock:
            if not due:
                due = (now - self._last_checked_monotonic) >= self.poll_interval_seconds
            if due and not self._refresh_inflight:
                self._refresh_inflight = True
                can_start = True
        if due and can_start:
            threading.Thread(target=self._refresh_once, daemon=True).start()
        with self._lock:
            return dict(self._state)

    def _set_state(self, state: dict[str, Any]) -> None:
        with self._lock:
            self._state = state
            self._last_checked_monotonic = time.monotonic()
            self._refresh_inflight = False

    def _refresh_once(self) -> None:
        home = Path.home()
        checked_at = datetime.now(timezone.utc).isoformat()

        try:
            auth_type = self._read_selected_auth_type(home)
            if auth_type in {"api-key", "vertex-ai"}:
                self._set_state(
                    {
                        "available": False,
                        "checked": True,
                        "checked_at": checked_at,
                        "auth_type": auth_type,
                        "account_email": "",
                        "tier": "",
                        "project_id": "",
                        "models": [],
                        "lowest_percent_left": None,
                        "last_error": "unsupported_auth_type",
                        "guidance": "Gemini quota monitoring requires OAuth auth mode.",
                        "next_steps": [
                            "Set Gemini CLI auth mode to oauth-personal.",
                            "Sign in with Gemini CLI in this environment.",
                        ],
                    }
                )
                return

            creds = self._load_oauth_creds(home)
            access_token = str(creds.get("access_token") or "")
            refresh_token = str(creds.get("refresh_token") or "")
            id_token = str(creds.get("id_token") or "")
            expiry_date_ms = creds.get("expiry_date")

            if not access_token:
                raise RuntimeError("missing_access_token")

            if isinstance(expiry_date_ms, (int, float)):
                expires_at = datetime.fromtimestamp(float(expiry_date_ms) / 1000, tz=timezone.utc)
                if expires_at <= datetime.now(timezone.utc) and refresh_token:
                    client_id, client_secret = self._extract_client_credentials()
                    if client_id and client_secret:
                        refreshed = self._refresh_access_token(
                            refresh_token=refresh_token,
                            client_id=client_id,
                            client_secret=client_secret,
                        )
                        access_token = str(refreshed.get("access_token") or access_token)
                        if access_token:
                            creds["access_token"] = access_token
                            expires_in = refreshed.get("expires_in")
                            if isinstance(expires_in, (int, float)):
                                new_expiry = datetime.now(timezone.utc) + timedelta(seconds=float(expires_in))
                                creds["expiry_date"] = int(new_expiry.timestamp() * 1000)
                            if refreshed.get("id_token"):
                                id_token = str(refreshed["id_token"])
                                creds["id_token"] = id_token
                            try:
                                creds_path = home / ".gemini" / "oauth_creds.json"
                                creds_path.write_text(json.dumps(creds, indent=2), encoding="utf-8")
                            except OSError:
                                pass

            account_email, _hosted_domain = self._parse_id_token_claims(id_token)
            tier, project_id = self._load_code_assist_status(access_token)
            models_payload = self._retrieve_user_quota(access_token, project_id)
            models, lowest = self._summarize_quota_buckets(models_payload)

            self._set_state(
                {
                    "available": True,
                    "checked": True,
                    "checked_at": checked_at,
                    "auth_type": auth_type,
                    "account_email": account_email,
                    "tier": tier,
                    "project_id": project_id,
                    "models": models,
                    "lowest_percent_left": lowest,
                    "last_error": "",
                    "guidance": "Gemini OAuth-backed quota monitoring is active.",
                    "next_steps": [],
                }
            )
            return

        except FileNotFoundError:
            self._set_state(
                {
                    "available": False,
                    "checked": True,
                    "checked_at": checked_at,
                    "auth_type": "unknown",
                    "account_email": "",
                    "tier": "",
                    "project_id": "",
                    "models": [],
                    "lowest_percent_left": None,
                    "last_error": "oauth_creds_not_found",
                    "guidance": "Gemini OAuth credentials were not found.",
                    "next_steps": [
                        "Sign in with Gemini CLI in this shell environment.",
                        "Re-run context.index_inspection after sign-in.",
                    ],
                }
            )
            return
        except RuntimeError as exc:
            code = str(exc)
            if code == "missing_access_token":
                guidance = "Gemini OAuth credentials are present but missing access token."
                steps = [
                    "Re-authenticate with Gemini CLI in this environment.",
                    "Re-run context.index_inspection after sign-in.",
                ]
            else:
                guidance = "Gemini quota monitoring failed while using OAuth credentials."
                steps = [
                    "Check Gemini CLI auth and network access in this environment.",
                    "Re-run context.index_inspection to retry quota polling.",
                ]
            self._set_state(
                {
                    "available": False,
                    "checked": True,
                    "checked_at": checked_at,
                    "auth_type": "oauth-personal",
                    "account_email": "",
                    "tier": "",
                    "project_id": "",
                    "models": [],
                    "lowest_percent_left": None,
                    "last_error": code,
                    "guidance": guidance,
                    "next_steps": steps,
                }
            )
            return
        except Exception as exc:  # pragma: no cover - defensive fallback
            self._set_state(
                {
                    "available": False,
                    "checked": True,
                    "checked_at": checked_at,
                    "auth_type": "unknown",
                    "account_email": "",
                    "tier": "",
                    "project_id": "",
                    "models": [],
                    "lowest_percent_left": None,
                    "last_error": f"quota_poll_error: {type(exc).__name__}",
                    "guidance": "Gemini quota monitor encountered an unexpected error.",
                    "next_steps": [
                        "Confirm Gemini CLI OAuth works in this environment.",
                        "Retry context.index_inspection.",
                    ],
                }
            )

    @staticmethod
    def _read_selected_auth_type(home: Path) -> str:
        settings_path = home / ".gemini" / "settings.json"
        if not settings_path.exists():
            return "unknown"
        try:
            raw = json.loads(settings_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "unknown"
        security = raw.get("security") if isinstance(raw, dict) else None
        auth = security.get("auth") if isinstance(security, dict) else None
        selected = auth.get("selectedType") if isinstance(auth, dict) else None
        if isinstance(selected, str) and selected:
            return selected
        return "unknown"

    @staticmethod
    def _load_oauth_creds(home: Path) -> dict[str, Any]:
        creds_path = home / ".gemini" / "oauth_creds.json"
        payload = json.loads(creds_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("invalid_oauth_creds")
        return payload

    @staticmethod
    def _parse_id_token_claims(id_token: str) -> tuple[str, str]:
        if not id_token:
            return "", ""
        parts = id_token.split(".")
        if len(parts) < 2:
            return "", ""
        payload_segment = parts[1]
        payload_segment += "=" * ((4 - len(payload_segment) % 4) % 4)
        try:
            decoded = base64.urlsafe_b64decode(payload_segment.encode("utf-8")).decode("utf-8")
            payload = json.loads(decoded)
            if not isinstance(payload, dict):
                return "", ""
            email = payload.get("email") if isinstance(payload.get("email"), str) else ""
            hosted_domain = payload.get("hd") if isinstance(payload.get("hd"), str) else ""
            return email, hosted_domain
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
            return "", ""

    def _extract_client_credentials(self) -> tuple[str, str]:
        binary = shutil.which(self.command)
        if not binary:
            return "", ""

        resolved = Path(binary).resolve()
        bin_dir = resolved.parent
        base_dir = bin_dir.parent
        candidates = [
            base_dir / "libexec/lib/node_modules/@google/gemini-cli/node_modules/@google/gemini-cli-core/dist/src/code_assist/oauth2.js",
            base_dir / "lib/node_modules/@google/gemini-cli/node_modules/@google/gemini-cli-core/dist/src/code_assist/oauth2.js",
            base_dir / "node_modules/@google/gemini-cli-core/dist/src/code_assist/oauth2.js",
            base_dir / "share/gemini-cli/node_modules/@google/gemini-cli-core/dist/src/code_assist/oauth2.js",
            base_dir.parent / "gemini-cli-core/dist/src/code_assist/oauth2.js",
        ]

        client_id_re = re.compile(r"OAUTH_CLIENT_ID\\s*=\\s*['\"]([^'\"]+)['\"]")
        client_secret_re = re.compile(r"OAUTH_CLIENT_SECRET\\s*=\\s*['\"]([^'\"]+)['\"]")

        for path in candidates:
            if not path.exists():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            id_match = client_id_re.search(text)
            secret_match = client_secret_re.search(text)
            if id_match and secret_match:
                return id_match.group(1), secret_match.group(1)

        return "", ""

    def _refresh_access_token(self, *, refresh_token: str, client_id: str, client_secret: str) -> dict[str, Any]:
        form = urllib.parse.urlencode(
            {
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            self._TOKEN_REFRESH_URL,
            data=form,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if not isinstance(payload, dict) or not payload.get("access_token"):
            raise RuntimeError("token_refresh_failed")
        return payload

    def _post_json(self, *, url: str, bearer_token: str, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {bearer_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("invalid_api_response")
        return payload

    def _load_code_assist_status(self, access_token: str) -> tuple[str, str]:
        try:
            payload = self._post_json(
                url=self._LOAD_CODE_ASSIST_URL,
                bearer_token=access_token,
                body={"metadata": {"ideType": "GEMINI_CLI", "pluginType": "GEMINI"}},
            )
        except Exception:
            return "", ""

        tier = ""
        current_tier = payload.get("currentTier")
        if isinstance(current_tier, dict):
            tier_id = current_tier.get("id")
            if isinstance(tier_id, str):
                tier = tier_id

        project_id = ""
        project = payload.get("cloudaicompanionProject")
        if isinstance(project, str):
            project_id = project.strip()
        elif isinstance(project, dict):
            candidate = project.get("id") or project.get("projectId")
            if isinstance(candidate, str):
                project_id = candidate.strip()

        return tier, project_id

    def _retrieve_user_quota(self, access_token: str, project_id: str) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if project_id:
            body["project"] = project_id
        return self._post_json(
            url=self._RETRIEVE_USER_QUOTA_URL,
            bearer_token=access_token,
            body=body,
        )

    @staticmethod
    def _summarize_quota_buckets(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], float | None]:
        buckets = payload.get("buckets") if isinstance(payload, dict) else None
        if not isinstance(buckets, list):
            return [], None

        by_model: dict[str, dict[str, Any]] = {}
        for bucket in buckets:
            if not isinstance(bucket, dict):
                continue
            model_id = bucket.get("modelId")
            remaining_fraction = bucket.get("remainingFraction")
            if not isinstance(model_id, str):
                continue
            if not isinstance(remaining_fraction, (int, float)):
                continue
            percent_left = float(remaining_fraction) * 100.0
            existing = by_model.get(model_id)
            if existing is None or percent_left < float(existing.get("percent_left", 100.0)):
                reset_time = bucket.get("resetTime") if isinstance(bucket.get("resetTime"), str) else ""
                by_model[model_id] = {
                    "model_id": model_id,
                    "percent_left": round(percent_left, 3),
                    "reset_time": reset_time,
                }

        models = [by_model[k] for k in sorted(by_model.keys())]
        if not models:
            return [], None

        lowest = min(float(item["percent_left"]) for item in models)
        return models, round(lowest, 3)


class GeminiCliStatsMonitor:
    _ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    _MODEL_LINE_RE = re.compile(
        r"(gemini-[\w\.-]+)\s+([0-9]+|-)\s+([0-9]+(?:\.[0-9]+)?)%\s+\(Resets in ([^)]+)\)"
    )
    _TOOL_CALLS_RE = re.compile(r"Tool Calls:\s*(\d+)")
    _SUCCESS_RE = re.compile(r"Success Rate:\s*([0-9]+(?:\.[0-9]+)?)%")
    _SESSION_RE = re.compile(r"Session ID:\s*([0-9a-f-]{12,})", re.IGNORECASE)

    def __init__(
        self,
        *,
        command: str,
        args: list[str],
        timeout_seconds: int,
        poll_interval_seconds: int,
    ) -> None:
        self.command = command
        self.args = list(args)
        self.timeout_seconds = max(5, min(timeout_seconds, 25))
        self.poll_interval_seconds = max(30, poll_interval_seconds)
        self._lock = threading.Lock()
        self._last_checked_monotonic = 0.0
        self._refresh_inflight = False
        self._state: dict[str, Any] = {
            "available": False,
            "checked": False,
            "checked_at": "",
            "session_id": "",
            "session_model": "",
            "tool_calls": 0,
            "success_rate_percent": None,
            "models": [],
            "lowest_percent_left": None,
            "last_error": "not_checked",
            "guidance": "Gemini CLI /stats has not been checked yet.",
            "next_steps": [
                "Run context.index_inspection to trigger CLI /stats polling.",
            ],
        }

    def status(self, force: bool = False, allow_poll: bool = True) -> dict[str, Any]:
        if not allow_poll:
            with self._lock:
                return dict(self._state)
        if force:
            run_sync = False
            with self._lock:
                if not self._refresh_inflight:
                    self._refresh_inflight = True
                    run_sync = True
            if run_sync:
                self._refresh_once()
            else:
                deadline = time.monotonic() + float(self.timeout_seconds + 1)
                while time.monotonic() < deadline:
                    with self._lock:
                        if not self._refresh_inflight:
                            break
                    time.sleep(0.05)
            with self._lock:
                return dict(self._state)
        due = force
        now = time.monotonic()
        can_start = False
        with self._lock:
            if not due:
                due = (now - self._last_checked_monotonic) >= self.poll_interval_seconds
            if due and not self._refresh_inflight:
                self._refresh_inflight = True
                can_start = True
        if due and can_start:
            threading.Thread(target=self._refresh_once, daemon=True).start()
        with self._lock:
            return dict(self._state)

    def _set_state(self, state: dict[str, Any]) -> None:
        with self._lock:
            self._state = state
            self._last_checked_monotonic = time.monotonic()
            self._refresh_inflight = False

    def _build_command(self) -> list[str]:
        cmd = [self.command, *self.args]
        if (
            "--output-format" not in self.args
            and "-o" not in self.args
            and not any(arg.startswith("--output-format=") for arg in self.args)
        ):
            cmd.extend(["--output-format", "text"])
        cmd.extend(["-p", "/stats"])
        return cmd

    def _clean_text(self, raw: str) -> str:
        text = self._ANSI_RE.sub("", raw)
        text = text.replace("\r", "\n")
        lines: list[str] = []
        for line in text.splitlines():
            normalized = re.sub(r"[│┃╎]", " ", line)
            stripped = normalized.strip()
            if not stripped:
                continue
            if re.match(r"^[\-\u2500-\u257F\s]+$", stripped):
                continue
            if stripped.startswith("Please visit the following URL to authorize"):
                continue
            if stripped.startswith("Enter the authorization code"):
                continue
            if stripped.startswith("https://accounts.google.com/o/oauth2"):
                continue
            lines.append(stripped)
        return "\n".join(lines)

    def _parse_stats(self, raw: str) -> dict[str, Any]:
        text = self._clean_text(raw)
        models: list[dict[str, Any]] = []
        for line in text.splitlines():
            m = self._MODEL_LINE_RE.match(line)
            if not m:
                continue
            model_id = m.group(1)
            reqs_raw = m.group(2)
            reqs = int(reqs_raw) if reqs_raw.isdigit() else None
            percent_left = float(m.group(3))
            reset_hint = m.group(4).strip()
            models.append(
                {
                    "model_id": model_id,
                    "requests": reqs,
                    "percent_left": round(percent_left, 3),
                    "reset_hint": reset_hint,
                }
            )

        if not models:
            raise RuntimeError("model_lines_not_found")

        lowest = min(item["percent_left"] for item in models)
        tool_calls = 0
        success_rate = None
        session_id = ""
        session_model = ""

        m_tools = self._TOOL_CALLS_RE.search(text)
        if m_tools:
            tool_calls = int(m_tools.group(1))
        m_success = self._SUCCESS_RE.search(text)
        if m_success:
            success_rate = float(m_success.group(1))
        m_session = self._SESSION_RE.search(text)
        if m_session:
            session_id = m_session.group(1)

        with_reqs = [item for item in models if isinstance(item.get("requests"), int) and int(item.get("requests", 0)) > 0]
        if with_reqs:
            with_reqs.sort(key=lambda item: int(item.get("requests", 0)), reverse=True)
            session_model = str(with_reqs[0].get("model_id", ""))
        else:
            fallback = next((item for item in models if "flash" in str(item.get("model_id", "")).lower()), models[0])
            session_model = str(fallback.get("model_id", ""))

        return {
            "session_id": session_id,
            "session_model": session_model,
            "tool_calls": tool_calls,
            "success_rate_percent": success_rate,
            "models": sorted(models, key=lambda item: item["model_id"]),
            "lowest_percent_left": round(lowest, 3),
        }

    def _refresh_once(self) -> None:
        checked_at = datetime.now(timezone.utc).isoformat()
        cmd = self._build_command()
        env = os.environ.copy()
        env.setdefault("NO_BROWSER", "true")
        env.setdefault("GEMINI_DEFAULT_AUTH_TYPE", "oauth-personal")
        env.setdefault("OAUTH_CALLBACK_HOST", "127.0.0.1")
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
        except OSError:
            self._set_state(
                {
                    "available": False,
                    "checked": True,
                    "checked_at": checked_at,
                    "session_id": "",
                    "session_model": "",
                    "tool_calls": 0,
                    "success_rate_percent": None,
                    "models": [],
                    "lowest_percent_left": None,
                    "last_error": "cli_not_found_or_not_executable",
                    "guidance": "Gemini CLI /stats command is not executable.",
                    "next_steps": [
                        "Ensure Gemini CLI is installed and available on PATH.",
                    ],
                }
            )
            return

        try:
            stdout, stderr = proc.communicate("", timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            self._set_state(
                {
                    "available": False,
                    "checked": True,
                    "checked_at": checked_at,
                    "session_id": "",
                    "session_model": "",
                    "tool_calls": 0,
                    "success_rate_percent": None,
                    "models": [],
                    "lowest_percent_left": None,
                    "last_error": "timeout",
                    "guidance": "Gemini CLI /stats timed out.",
                    "next_steps": [
                        "Verify Gemini CLI is authenticated and not waiting for input.",
                    ],
                }
            )
            return

        if proc.returncode != 0:
            message = (stderr or "unknown error").strip().replace("\n", " ")
            self._set_state(
                {
                    "available": False,
                    "checked": True,
                    "checked_at": checked_at,
                    "session_id": "",
                    "session_model": "",
                    "tool_calls": 0,
                    "success_rate_percent": None,
                    "models": [],
                    "lowest_percent_left": None,
                    "last_error": f"cli_error: {message[:200]}",
                    "guidance": "Gemini CLI returned an error for /stats.",
                    "next_steps": [
                        "Run `gemini -p \"/stats\"` in the same environment and confirm it succeeds.",
                    ],
                }
            )
            return

        try:
            parsed = self._parse_stats(stdout)
        except RuntimeError:
            self._set_state(
                {
                    "available": False,
                    "checked": True,
                    "checked_at": checked_at,
                    "session_id": "",
                    "session_model": "",
                    "tool_calls": 0,
                    "success_rate_percent": None,
                    "models": [],
                    "lowest_percent_left": None,
                    "last_error": "stats_parse_failed",
                    "guidance": "Gemini CLI /stats output could not be parsed in headless mode.",
                    "next_steps": [
                        "Confirm `gemini -p \"/stats\"` prints model usage rows in this environment.",
                    ],
                }
            )
            return

        self._set_state(
            {
                "available": True,
                "checked": True,
                "checked_at": checked_at,
                "session_id": parsed["session_id"],
                "session_model": parsed["session_model"],
                "tool_calls": parsed["tool_calls"],
                "success_rate_percent": parsed["success_rate_percent"],
                "models": parsed["models"],
                "lowest_percent_left": parsed["lowest_percent_left"],
                "last_error": "",
                "guidance": "Gemini CLI /stats polling is active.",
                "next_steps": [],
            }
        )


class GeminiReranker:
    def __init__(self, config: GeminiConfig) -> None:
        self.config = config
        self._connection_checked = False
        self._connected = False
        self._last_error = "not_checked"
        self._last_error_detail = ""
        self._quota_monitor = GeminiQuotaMonitor(
            command=config.command,
            timeout_seconds=config.timeout_seconds,
            poll_interval_seconds=config.quota_poll_seconds,
        )
        self._cli_stats_monitor = GeminiCliStatsMonitor(
            command=config.command,
            args=list(config.args),
            timeout_seconds=config.timeout_seconds,
            poll_interval_seconds=config.quota_poll_seconds,
        )
        self._usage = {
            "requests_total": 0,
            "cli_invocations_total": 0,
            "cli_success_total": 0,
            "fallback_total": 0,
            "gemini_total_tokens": 0,
            "gemini_prompt_tokens": 0,
            "gemini_output_tokens": 0,
            "gemini_input_tokens": 0,
            "gemini_cached_tokens": 0,
            "gemini_thought_tokens": 0,
            "gemini_tool_tokens": 0,
            "last_call_tokens": {
                "total": 0,
                "prompt": 0,
                "output": 0,
                "input": 0,
                "cached": 0,
                "thought": 0,
                "tool": 0,
            },
            "last_request_at": "",
            "last_success_at": "",
            "last_usage_update_at": "",
        }
        self._plan_cache_lock = threading.Lock()
        self._plan_cache: dict[str, tuple[float, dict[str, Any]]] = {}

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _build_command(self) -> list[str]:
        cmd = [self.config.command, *self.config.args]
        if (
            "--output-format" not in self.config.args
            and "-o" not in self.config.args
            and not any(arg.startswith("--output-format=") for arg in self.config.args)
        ):
            cmd.extend(["--output-format", "json"])
        cmd.extend(["-p", ""])
        return cmd

    def _decode_json_payload(self, raw: str) -> Any:
        text = (raw or "").strip()
        if not text:
            raise ValueError("invalid_json_from_cli")

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Handle markdown-fenced JSON payloads that some CLI versions return.
        fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
        for block in fenced_blocks:
            block_text = block.strip()
            if not block_text:
                continue
            try:
                return json.loads(block_text)
            except json.JSONDecodeError:
                continue

        # Handle text that contains JSON with leading/trailing non-JSON lines.
        decoder = json.JSONDecoder()
        for marker in ("{", "["):
            start = text.find(marker)
            while start != -1:
                try:
                    obj, end = decoder.raw_decode(text[start:])
                except json.JSONDecodeError:
                    start = text.find(marker, start + 1)
                    continue
                if text[start + end :].strip():
                    start = text.find(marker, start + 1)
                    continue
                return obj
        raise ValueError("invalid_json_from_cli")

    def _parse_cli_output(self, stdout: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
        decoded = self._decode_json_payload(stdout)

        # Non-interactive Gemini JSON output is an envelope:
        # { "response": "<model text>", "stats": { ... } }
        if isinstance(decoded, dict) and "response" in decoded:
            stats = decoded.get("stats")
            stats_dict = stats if isinstance(stats, dict) else None
            response = decoded.get("response")
            if isinstance(response, str):
                payload = self._decode_json_payload(response)
            elif isinstance(response, dict):
                payload = response
            else:
                raise ValueError("invalid_json_from_cli")
            if not isinstance(payload, dict):
                raise ValueError("invalid_json_from_cli")
            return payload, stats_dict

        if isinstance(decoded, dict):
            return decoded, None

        raise ValueError("invalid_json_from_cli")

    def _extract_token_counts_from_stats(self, stats: dict[str, Any]) -> dict[str, int]:
        totals = {
            "total": 0,
            "prompt": 0,
            "output": 0,
            "input": 0,
            "cached": 0,
            "thought": 0,
            "tool": 0,
        }

        models = stats.get("models")
        if not isinstance(models, dict):
            return totals

        for model_metrics in models.values():
            if not isinstance(model_metrics, dict):
                continue
            token_metrics = model_metrics.get("tokens")
            if not isinstance(token_metrics, dict):
                continue
            totals["total"] += int(token_metrics.get("total", 0) or 0)
            totals["prompt"] += int(token_metrics.get("prompt", 0) or 0)
            totals["output"] += int(token_metrics.get("candidates", 0) or 0)
            totals["input"] += int(token_metrics.get("input", 0) or 0)
            totals["cached"] += int(token_metrics.get("cached", 0) or 0)
            totals["thought"] += int(token_metrics.get("thoughts", 0) or 0)
            totals["tool"] += int(token_metrics.get("tool", 0) or 0)
        return totals

    def _update_usage_from_stats(self, stats: dict[str, Any] | None) -> None:
        if not isinstance(stats, dict):
            return

        tokens = self._extract_token_counts_from_stats(stats)
        self._usage["gemini_total_tokens"] += tokens["total"]
        self._usage["gemini_prompt_tokens"] += tokens["prompt"]
        self._usage["gemini_output_tokens"] += tokens["output"]
        self._usage["gemini_input_tokens"] += tokens["input"]
        self._usage["gemini_cached_tokens"] += tokens["cached"]
        self._usage["gemini_thought_tokens"] += tokens["thought"]
        self._usage["gemini_tool_tokens"] += tokens["tool"]
        self._usage["last_call_tokens"] = dict(tokens)
        self._usage["last_usage_update_at"] = self._now_iso()

    def _build_prompt(self, query: str, candidates: list[dict], max_results: int) -> str:
        serialized = json.dumps(self._compact_candidates_for_prompt(candidates, max_results=max_results), ensure_ascii=True)
        query_text = self._compact_query_text(query, max_terms=14, max_chars=180)
        return (
            "Rank local code excerpts for relevance.\n"
            "Return JSON only using: {\"snippets\": [{\"id\": int, \"score\": float, \"rationale\": string}]}.\n"
            "Use only provided candidates. Do not invent code or files.\n"
            "Candidate keys: i=id, p=path_tail, b=base_score, e=excerpt.\n"
            "Set snippets[].id equal to candidate i.\n"
            f"Limit to at most {max_results} snippets.\n"
            f"Query terms: {query_text}\n"
            f"Candidates: {serialized}\n"
        )

    def _normalize_prompt_excerpt(self, excerpt: str) -> str:
        lines = str(excerpt).splitlines()
        if not lines:
            return ""

        # Trim trailing whitespace and drop outer blank lines.
        cleaned = [line.rstrip() for line in lines]
        while cleaned and not cleaned[0].strip():
            cleaned.pop(0)
        while cleaned and not cleaned[-1].strip():
            cleaned.pop()

        # Collapse repeated blank lines.
        collapsed: list[str] = []
        blank_open = False
        for line in cleaned:
            if not line.strip():
                if blank_open:
                    continue
                blank_open = True
                collapsed.append("")
                continue
            blank_open = False
            collapsed.append(line)

        # Remove shared left indentation (preserve relative indentation).
        indents = [len(line) - len(line.lstrip(" ")) for line in collapsed if line]
        shared = min(indents) if indents else 0
        shared = min(shared, 8)
        if shared > 0:
            collapsed = [line[shared:] if line and len(line) >= shared else line for line in collapsed]

        return "\n".join(collapsed)

    def _compact_candidates_for_prompt(self, candidates: list[dict], *, max_results: int) -> list[dict]:
        compact: list[dict] = []
        max_excerpt_chars = max(120, int(self.config.rerank_excerpt_chars))
        max_candidates = min(max(1, int(self.config.rerank_max_candidates)), max(max_results * 2, 8))
        prompt_char_budget = max(2000, int(self.config.rerank_prompt_char_budget))
        used_chars = 0

        ranked = sorted(candidates, key=lambda item: float(item.get("base_score", 0.0)), reverse=True)
        for item in ranked:
            if len(compact) >= max_candidates:
                break
            excerpt = self._normalize_prompt_excerpt(str(item.get("excerpt", "")))
            if len(excerpt) > max_excerpt_chars:
                excerpt = excerpt[:max_excerpt_chars] + "\n...[truncated]"
            remaining_budget = prompt_char_budget - used_chars
            if remaining_budget <= 0 and len(compact) >= max_results:
                break
            if len(excerpt) > remaining_budget and len(compact) >= max_results:
                break
            if len(excerpt) > remaining_budget:
                keep = max(40, min(max_excerpt_chars, max(0, remaining_budget)))
                excerpt = excerpt[:keep] + "\n...[truncated]"
            compact.append(
                {
                    "i": int(item.get("id", 0)),
                    "p": self._compact_path_for_prompt(str(item.get("path", ""))),
                    "e": excerpt,
                    "b": round(float(item.get("base_score", 0.0)), 6),
                }
            )
            used_chars += len(excerpt) + 64

        if not compact and ranked:
            first = ranked[0]
            compact.append(
                {
                    "i": int(first.get("id", 0)),
                    "p": self._compact_path_for_prompt(str(first.get("path", ""))),
                    "e": self._normalize_prompt_excerpt(str(first.get("excerpt", "")))[:max_excerpt_chars],
                    "b": round(float(first.get("base_score", 0.0)), 6),
                }
            )
        return compact

    def _compact_path_for_prompt(self, path: str) -> str:
        parts = [part for part in str(path).split("/") if part]
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        return "/".join(parts[-2:])

    def _build_retrieval_plan_prompt(
        self,
        *,
        query: str,
        available_paths: list[str],
        max_paths: int,
        max_terms: int,
        tool_name: str,
        hint_paths: list[str],
        hint_terms: list[str],
    ) -> str:
        catalog = json.dumps(available_paths, ensure_ascii=True)
        hint_paths_json = json.dumps(hint_paths[: max_paths * 2], ensure_ascii=True)
        hint_terms_json = json.dumps(hint_terms[: max_terms * 2], ensure_ascii=True)
        return (
            "Plan code retrieval for a local context engine.\n"
            "Return JSON only: {\"paths\": [string], \"terms\": [string], \"rationale\": string}.\n"
            "Use only provided catalog paths. No invented paths.\n"
            f"Tool: {tool_name}\n"
            f"Limit paths to at most {max_paths} and terms to at most {max_terms}.\n"
            f"Query terms: {query}\n"
            f"Path hints: {hint_paths_json}\n"
            f"Term hints: {hint_terms_json}\n"
            f"Path catalog: {catalog}\n"
        )

    def _select_planning_catalog(self, available_paths: list[str], query: str, max_catalog: int = 120) -> list[str]:
        if len(available_paths) <= max_catalog:
            return available_paths
        terms = self._fallback_terms(query, max_terms=8)
        scored: list[tuple[int, int, str]] = []
        for path in available_paths:
            lower = path.lower()
            score = sum(1 for term in terms if term and term in lower)
            if score > 0:
                scored.append((score, -len(path), path))
        scored.sort(reverse=True)
        selected: list[str] = []
        seen: set[str] = set()
        for _, _, path in scored:
            if path in seen:
                continue
            seen.add(path)
            selected.append(path)
            if len(selected) >= max_catalog:
                break
        if len(selected) < max_catalog:
            for path in available_paths:
                if path in seen:
                    continue
                seen.add(path)
                selected.append(path)
                if len(selected) >= max_catalog:
                    break
        return selected

    def _plan_cache_key(
        self,
        *,
        query: str,
        catalog: list[str],
        max_paths: int,
        max_terms: int,
    ) -> str:
        digest = hashlib.sha1("\n".join(catalog).encode("utf-8")).hexdigest()[:16]
        return f"{query.strip().lower()}|{len(catalog)}|{digest}|{max_paths}|{max_terms}"

    def _fallback_terms(self, query: str, max_terms: int) -> list[str]:
        terms = re.findall(r"[A-Za-z0-9_]{2,}", query.lower())
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

    def _compact_query_text(self, query: str, *, max_terms: int, max_chars: int) -> str:
        terms = self._fallback_terms(query, max_terms=max_terms)
        if terms:
            text = " ".join(terms)
        else:
            text = " ".join(str(query).strip().split())
        text = text[:max_chars].strip()
        return text or "context"

    def _extract_path_hints(self, query: str) -> list[str]:
        # File/path hints like "foo.py", "src/module.swift", "a/b/c.tsx".
        matches = re.findall(r"(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.[A-Za-z0-9_+-]+", query or "")
        hints: list[str] = []
        seen: set[str] = set()
        for match in matches:
            cleaned = match.strip().strip("\"'").lstrip("./")
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            hints.append(cleaned)
            if len(hints) >= 20:
                break
        # Extensionless symbol/file hints, e.g. "MultiLaneTimelineView".
        symbol_tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]{5,}", query or "")
        for token in symbol_tokens:
            if not any(char.isupper() for char in token[1:]) and "_" not in token:
                continue
            cleaned = token.strip().strip("\"'")
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            hints.append(cleaned)
            if len(hints) >= 30:
                break
        return hints

    def _resolve_hint_paths(
        self,
        *,
        available_paths: list[str],
        hint_paths: list[str],
        max_paths: int,
    ) -> list[str]:
        if not available_paths or not hint_paths:
            return []

        by_lower = {path.lower(): path for path in available_paths}
        basename_map: dict[str, list[str]] = {}
        stem_map: dict[str, list[str]] = {}
        for path in available_paths:
            basename = path.rsplit("/", 1)[-1].lower()
            basename_map.setdefault(basename, []).append(path)
            stem = re.sub(r"[^a-z0-9]+", "", basename.rsplit(".", 1)[0])
            if stem:
                stem_map.setdefault(stem, []).append(path)

        resolved: list[str] = []
        seen: set[str] = set()
        for raw_hint in hint_paths:
            hint = raw_hint.strip().strip("\"'").lstrip("./").lower()
            if not hint:
                continue

            exact = by_lower.get(hint)
            if exact and exact not in seen:
                seen.add(exact)
                resolved.append(exact)
                if len(resolved) >= max_paths:
                    break

            basename = hint.rsplit("/", 1)[-1]
            for candidate in basename_map.get(basename, []):
                if candidate in seen:
                    continue
                seen.add(candidate)
                resolved.append(candidate)
                if len(resolved) >= max_paths:
                    break
            if len(resolved) >= max_paths:
                break

            normalized_hint = re.sub(r"[^a-z0-9]+", "", basename)
            for candidate in stem_map.get(normalized_hint, []):
                if candidate in seen:
                    continue
                seen.add(candidate)
                resolved.append(candidate)
                if len(resolved) >= max_paths:
                    break
            if len(resolved) >= max_paths:
                break

            for candidate in available_paths:
                lowered = candidate.lower()
                if hint not in lowered:
                    continue
                if candidate in seen:
                    continue
                seen.add(candidate)
                resolved.append(candidate)
                if len(resolved) >= max_paths:
                    break
            if len(resolved) >= max_paths:
                break
        return resolved

    def _dedupe_terms(self, values: list[str], *, max_terms: int) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            term = value.strip()
            if not term:
                continue
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(term)
            if len(deduped) >= max_terms:
                break
        return deduped

    def plan_retrieval(
        self,
        *,
        query: str,
        available_paths: list[str],
        max_paths: int,
        max_terms: int,
        tool_name: str = "context.retrieval",
        hint_paths: list[str] | None = None,
        hint_terms: list[str] | None = None,
    ) -> dict[str, Any]:
        bounded_paths = max(1, max_paths)
        bounded_terms = max(1, max_terms)
        fallback_terms = self._fallback_terms(query, bounded_terms)
        strategy = str(self.config.planning_strategy or "embedded").strip().lower()
        if not available_paths:
            return {
                "paths": [],
                "terms": fallback_terms,
                "rationale": "No indexed paths available; deterministic query-term fallback.",
                "source": "deterministic_fallback",
            }

        resolved_hint_paths = self._resolve_hint_paths(
            available_paths=available_paths,
            hint_paths=(hint_paths or []) + self._extract_path_hints(query),
            max_paths=bounded_paths,
        )
        resolved_hint_terms = self._dedupe_terms(
            [term for term in (hint_terms or []) if isinstance(term, str)] + fallback_terms,
            max_terms=bounded_terms,
        )
        if resolved_hint_paths and resolved_hint_terms:
            return {
                "paths": resolved_hint_paths,
                "terms": resolved_hint_terms,
                "rationale": "Embedded tool hints supplied explicit file/keyword guidance; planner call skipped.",
                "source": "embedded_hints",
            }

        if strategy in {"embedded", "hints", "deterministic"}:
            return {
                "paths": [],
                "terms": resolved_hint_terms or fallback_terms,
                "rationale": "Embedded planning strategy active; Gemini planner call skipped.",
                "source": "embedded_only",
            }

        if not self._connected:
            self.connect()
        if not self._connected:
            self._usage["fallback_total"] += 1
            return {
                "paths": [],
                "terms": fallback_terms,
                "rationale": "Gemini unavailable; deterministic query-term fallback.",
                "source": "deterministic_fallback",
            }

        compact_query = self._compact_query_text(query, max_terms=max(bounded_terms * 2, 8), max_chars=180)
        catalog = self._select_planning_catalog(
            available_paths,
            compact_query,
            max_catalog=max(10, int(self.config.planning_max_catalog_paths)),
        )
        cache_key = self._plan_cache_key(
            query=compact_query,
            catalog=catalog,
            max_paths=bounded_paths,
            max_terms=bounded_terms,
        )
        with self._plan_cache_lock:
            cached = self._plan_cache.get(cache_key)
        if cached is not None:
            cached_at, payload = cached
            if (time.monotonic() - cached_at) <= max(1, int(self.config.planning_cache_seconds)):
                return dict(payload)

        prompt = self._build_retrieval_plan_prompt(
            query=compact_query,
            available_paths=catalog,
            max_paths=bounded_paths,
            max_terms=bounded_terms,
            tool_name=tool_name,
            hint_paths=resolved_hint_paths,
            hint_terms=resolved_hint_terms,
        )
        result = self._execute(prompt)
        if result is None:
            self._usage["fallback_total"] += 1
            return {
                "paths": [],
                "terms": fallback_terms,
                "rationale": "Gemini execution failed; deterministic query-term fallback.",
                "source": "deterministic_fallback",
            }

        returncode, stdout, stderr = result
        if returncode != 0:
            self._connected = False
            message = (stderr or "unknown error").strip().replace("\n", " ")
            self._last_error = f"cli_error: {message[:240]}"
            self._last_error_detail = message[:1000]
            self._usage["fallback_total"] += 1
            return {
                "paths": [],
                "terms": fallback_terms,
                "rationale": "Gemini execution error; deterministic query-term fallback.",
                "source": "deterministic_fallback",
            }

        try:
            payload, stats = self._parse_cli_output(stdout)
            if not isinstance(payload, dict):
                raise ValueError("invalid_json_from_cli")
            self._update_usage_from_stats(stats)
            raw_paths = payload.get("paths", [])
            raw_terms = payload.get("terms", [])
            raw_rationale = payload.get("rationale", "")
            if not isinstance(raw_paths, list) or not isinstance(raw_terms, list):
                raise ValueError("invalid_json_from_cli")
        except (json.JSONDecodeError, ValueError, TypeError):
            self._connected = False
            self._last_error = "invalid_json_from_cli"
            self._last_error_detail = ""
            self._usage["fallback_total"] += 1
            return {
                "paths": [],
                "terms": fallback_terms,
                "rationale": "Gemini returned invalid plan JSON; deterministic query-term fallback.",
                "source": "deterministic_fallback",
            }

        valid_paths = set(available_paths)
        selected_paths: list[str] = []
        seen_paths: set[str] = set()
        for item in raw_paths:
            if not isinstance(item, str):
                continue
            path = item.strip()
            if not path or path in seen_paths or path not in valid_paths:
                continue
            seen_paths.add(path)
            selected_paths.append(path)
            if len(selected_paths) >= bounded_paths:
                break

        selected_terms: list[str] = []
        seen_terms: set[str] = set()
        for item in raw_terms:
            if not isinstance(item, str):
                continue
            term = item.strip()
            if not term:
                continue
            key = term.lower()
            if key in seen_terms:
                continue
            seen_terms.add(key)
            selected_terms.append(term)
            if len(selected_terms) >= bounded_terms:
                break

        if not selected_terms:
            selected_terms = fallback_terms

        rationale = raw_rationale.strip() if isinstance(raw_rationale, str) else ""
        plan = {
            "paths": selected_paths,
            "terms": selected_terms,
            "rationale": rationale or "Gemini retrieval plan generated.",
            "source": "gemini",
        }
        with self._plan_cache_lock:
            self._plan_cache[cache_key] = (time.monotonic(), dict(plan))
            if len(self._plan_cache) > 200:
                oldest_key = min(self._plan_cache.items(), key=lambda item: item[1][0])[0]
                self._plan_cache.pop(oldest_key, None)
        return plan

    def _deterministic_fallback(self, candidates: list[dict], max_results: int) -> list[dict]:
        ranked = sorted(candidates, key=lambda c: c.get("base_score", 0.0), reverse=True)
        return [
            {
                "id": item["id"],
                "score": float(item.get("base_score", 0.0)),
                "rationale": "Deterministic lexical-density fallback ranking.",
            }
            for item in ranked[:max_results]
        ]

    def _guidance(self) -> tuple[str, list[str]]:
        code = self._last_error
        command_hint = self.config.command

        if code == "not_checked":
            return (
                "Gemini CLI connection has not been checked yet.",
                ["Run context.relevant_code or context.index_inspection to trigger a connection check."],
            )
        if code == "cli_not_found_or_not_executable":
            return (
                "Gemini CLI was not found or cannot be executed.",
                [
                    f"Run `{command_hint} --help` in this same environment to verify the CLI is installed and on PATH.",
                    "If your executable name is different, set `gemini_command` in `config.yaml`.",
                    "After fixing the command, run another context request; restart is not required.",
                ],
            )
        if code == "timeout":
            return (
                "Gemini CLI timed out before returning a response.",
                [
                    "Run the Gemini CLI directly to ensure it is not waiting for interactive login.",
                    "Authenticate the Gemini CLI in this same shell environment if prompted.",
                    "Increase `gemini_timeout_seconds` in `config.yaml` if the local model call is slow.",
                    "After fixing authentication/timeout, run another context request; restart is not required.",
                ],
            )
        if code == "output_limit_exceeded":
            return (
                "Gemini CLI output exceeded the configured size limit.",
                [
                    "Reduce request size (`max_results` or `max_excerpt_lines`) for context.relevant_code.",
                    "Optionally increase `gemini_max_output_bytes` in `config.yaml`.",
                    "Run another context request after adjustment; restart is not required.",
                ],
            )
        if code == "invalid_json_from_cli":
            return (
                "Gemini CLI returned a non-JSON or invalid JSON response.",
                [
                    "Ensure your Gemini CLI invocation returns raw JSON to stdout.",
                    "Check `gemini_args` in `config.yaml` so no markdown/prose wrappers are added.",
                    "Run another context request after adjustment; restart is not required.",
                ],
            )
        if code == "auth_interactive_required":
            return (
                "Gemini CLI requires OAuth sign-in in this environment before non-interactive JSON calls can run.",
                [
                    "Run `gemini` once in a regular terminal and complete the OAuth flow.",
                    "Keep `GEMINI_DEFAULT_AUTH_TYPE=oauth-personal` and `NO_BROWSER=true` for MCP sessions.",
                    "After successful sign-in, re-run context.index_inspection.",
                ],
            )
        if code.startswith("cli_error"):
            detail = (self._last_error_detail or "").lower()
            if "listen eperm" in detail:
                return (
                    "Gemini OAuth callback listener is blocked in this environment.",
                    [
                        "Set `NO_BROWSER=true` for the MCP server environment to avoid localhost callback auth.",
                        "Run `NO_BROWSER=true gemini` once in a regular terminal to complete OAuth device-code login.",
                        "Keep `GEMINI_DEFAULT_AUTH_TYPE=oauth-personal` so OAuth remains the active auth mode.",
                        "After login succeeds once, run another context request; restart is not required.",
                    ],
                )
            return (
                "Gemini CLI returned an execution error.",
                [
                    "Run the same Gemini CLI command in this environment to see the full error.",
                    "If the error indicates authentication, sign in to Gemini CLI in this shell environment.",
                    "After fixing the CLI error, run another context request; restart is not required.",
                ],
            )
        return (
            "Gemini CLI is unavailable; deterministic fallback ranking is active.",
            ["Resolve Gemini CLI availability/authentication issues and retry a context request."],
        )

    def _execute(self, prompt: str) -> tuple[int, str, str] | None:
        # Always invoke Gemini in non-interactive mode so subprocess I/O is deterministic.
        # Keep prompt on stdin to avoid shell/argv length limits.
        cmd = self._build_command()
        self._usage["cli_invocations_total"] += 1

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError:
            self._connected = False
            self._last_error = "cli_not_found_or_not_executable"
            self._last_error_detail = ""
            self._connection_checked = True
            return None

        try:
            stdout, stderr = proc.communicate(prompt, timeout=self.config.timeout_seconds)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            self._connected = False
            self._last_error = "timeout"
            self._last_error_detail = ""
            self._connection_checked = True
            return None

        if len(stdout.encode("utf-8")) > self.config.max_output_bytes:
            self._connected = False
            self._last_error = "output_limit_exceeded"
            self._last_error_detail = ""
            self._connection_checked = True
            return None

        return proc.returncode, stdout, stderr

    def connect(self) -> dict[str, Any]:
        # lightweight no-op ranking check that validates local CLI session/auth.
        prompt = self._build_prompt(
            "connection probe",
            [{"id": 0, "path": "probe", "start_line": 1, "end_line": 1, "excerpt": "probe", "base_score": 1.0}],
            1,
        )
        result = self._execute(prompt)
        if result is None:
            return self.status()

        returncode, stdout, stderr = result
        if returncode != 0:
            self._connected = False
            message = (stderr or "unknown error").strip().replace("\n", " ")
            self._last_error = f"cli_error: {message[:240]}"
            self._last_error_detail = message[:1000]
            self._connection_checked = True
            return self.status()

        try:
            payload, stats = self._parse_cli_output(stdout)
            validate_gemini_rerank_payload(payload)
            self._update_usage_from_stats(stats)
        except (json.JSONDecodeError, ValueError):
            self._connected = False
            lower = stdout.lower()
            if "please visit the following url to authorize" in lower or "enter the authorization code" in lower:
                self._last_error = "auth_interactive_required"
                self._last_error_detail = stdout[:1000]
            else:
                self._last_error = "invalid_json_from_cli"
                self._last_error_detail = stdout[:1000]
            self._connection_checked = True
            return self.status()

        self._connected = True
        self._last_error = ""
        self._last_error_detail = ""
        self._connection_checked = True
        return self.status()

    def status(self, force: bool = False) -> dict[str, Any]:
        # Status checks should not trigger model invocations.
        # Connectivity probes happen on actual retrieval calls.
        guidance, steps = self._guidance() if not self._connected else ("Gemini CLI connected.", [])
        allow_monitor_poll = bool(force or self.config.auto_monitor_poll)
        return {
            "connected": self._connected,
            "checked": self._connection_checked,
            "last_error": self._last_error,
            "error_detail": self._last_error_detail,
            "guidance": guidance,
            "next_steps": steps,
            "fallback_active": not self._connected,
            "usage": dict(self._usage),
            "usage_notes": "Token values come from Gemini CLI JSON stats when available.",
            "quota": self._quota_monitor.status(force=force, allow_poll=allow_monitor_poll),
            "cli_stats": self._cli_stats_monitor.status(force=force, allow_poll=allow_monitor_poll),
        }

    def rerank(self, query: str, candidates: list[dict], max_results: int) -> list[dict]:
        if not candidates:
            return []

        self._usage["requests_total"] += 1
        self._usage["last_request_at"] = self._now_iso()

        # Auto-connect to local CLI session on first use or after a prior failure.
        if not self._connected:
            self.connect()
        if not self._connected:
            self._usage["fallback_total"] += 1
            return self._deterministic_fallback(candidates, max_results)

        prompt = self._build_prompt(query, candidates, max_results)
        result = self._execute(prompt)
        if result is None:
            self._usage["fallback_total"] += 1
            return self._deterministic_fallback(candidates, max_results)

        returncode, stdout, stderr = result
        if returncode != 0:
            self._connected = False
            message = (stderr or "unknown error").strip().replace("\n", " ")
            self._last_error = f"cli_error: {message[:240]}"
            self._last_error_detail = message[:1000]
            self._usage["fallback_total"] += 1
            return self._deterministic_fallback(candidates, max_results)

        try:
            payload, stats = self._parse_cli_output(stdout)
            normalized = validate_gemini_rerank_payload(payload)
            self._update_usage_from_stats(stats)
        except (json.JSONDecodeError, ValueError):
            self._connected = False
            lower = stdout.lower()
            if "please visit the following url to authorize" in lower or "enter the authorization code" in lower:
                self._last_error = "auth_interactive_required"
                self._last_error_detail = stdout[:1000]
            else:
                self._last_error = "invalid_json_from_cli"
                self._last_error_detail = stdout[:1000]
            self._usage["fallback_total"] += 1
            return self._deterministic_fallback(candidates, max_results)

        # Validate ids correspond to real candidates and keep first occurrence.
        candidate_ids = {item["id"] for item in candidates}
        seen: set[int] = set()
        filtered: list[dict] = []
        for item in normalized:
            cid = item["id"]
            if cid not in candidate_ids or cid in seen:
                continue
            seen.add(cid)
            filtered.append(item)
            if len(filtered) >= max_results:
                break

        if not filtered:
            self._usage["fallback_total"] += 1
            return self._deterministic_fallback(candidates, max_results)

        self._usage["cli_success_total"] += 1
        self._usage["last_success_at"] = self._now_iso()
        return filtered
