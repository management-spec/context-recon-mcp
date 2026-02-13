from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from browser_cookies import load_cookie_header


def _read_json(path: str) -> dict[str, Any] | None:
    try:
        data = Path(path).expanduser().read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        decoded = json.loads(data)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, dict) else None


def _extract_token(payload: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def load_claude_oauth_token() -> str:
    config_dir = Path.home() / ".claude"
    payload = _read_json(str(config_dir / ".credentials.json")) or {}
    return _extract_token(payload, ("access_token", "accessToken", "token"))


def load_codex_oauth_token() -> str:
    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    payload = _read_json(str(codex_home / "auth.json")) or {}
    return _extract_token(payload, ("access_token", "accessToken", "token"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fetch_json(url: str, *, headers: dict[str, str], timeout: int = 10) -> dict[str, Any] | None:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status >= 400:
                return None
            data = resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError):
        return None
    try:
        decoded = json.loads(data.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, dict) else None


def _fetch_text(url: str, *, headers: dict[str, str], timeout: int = 10) -> str | None:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status >= 400:
                return None
            data = resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError):
        return None
    return data.decode("utf-8", errors="replace")


def _percent_from_payload(payload: dict[str, Any]) -> float | None:
    for key in ("percent_used", "percentUsed", "used_percent", "usedPercent"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    for key in ("percent_left", "percentLeft"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return max(0.0, 100.0 - float(value))
    for key in ("remaining_fraction", "remainingFraction", "remainingFraction"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return max(0.0, 100.0 * (1.0 - float(value)))
    return None


def _reset_from_payload(payload: dict[str, Any]) -> str:
    for key in (
        "resets_at",
        "reset_at",
        "resetsAt",
        "resetAt",
        "reset_time",
        "resetTime",
        "resets_in",
        "resetsIn",
    ):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _window_payload(payload: dict[str, Any], label: str) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    percent = _percent_from_payload(payload)
    reset = _reset_from_payload(payload)
    if percent is None and not reset:
        return None
    return {
        "label": label,
        "percent_used": percent,
        "reset": reset,
    }


def _flatten_windows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    if not isinstance(payload, dict):
        return windows
    for key, label in (
        ("five_hour", "5h"),
        ("fiveHour", "5h"),
        ("session", "session"),
        ("seven_day", "7d"),
        ("sevenDay", "7d"),
        ("weekly", "weekly"),
        ("weekly_limit", "weekly"),
        ("opus", "opus"),
        ("sonnet", "sonnet"),
    ):
        raw = payload.get(key)
        if isinstance(raw, dict):
            window = _window_payload(raw, label)
            if window:
                windows.append(window)
    return windows


def fetch_claude_web_usage(*, cookie_header: str, cookie_source: str) -> dict[str, Any]:
    headers = {
        "Cookie": cookie_header,
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }
    orgs = _fetch_json("https://claude.ai/api/organizations", headers=headers)
    if not orgs:
        return {
            "available": False,
            "checked": True,
            "checked_at": _now_iso(),
            "source": "web",
            "cookie_source": cookie_source,
            "error": "orgs_not_found",
        }
    org_id = ""
    if isinstance(orgs.get("organization_uuid"), str):
        org_id = orgs["organization_uuid"]
    elif isinstance(orgs.get("organizations"), list) and orgs["organizations"]:
        candidate = orgs["organizations"][0]
        if isinstance(candidate, dict) and isinstance(candidate.get("uuid"), str):
            org_id = candidate["uuid"]
    if not org_id:
        return {
            "available": False,
            "checked": True,
            "checked_at": _now_iso(),
            "source": "web",
            "cookie_source": cookie_source,
            "error": "org_id_missing",
        }
    usage = _fetch_json(f"https://claude.ai/api/organizations/{org_id}/usage", headers=headers) or {}
    windows = _flatten_windows(usage)
    overage = _fetch_json(
        f"https://claude.ai/api/organizations/{org_id}/overage_spend_limit", headers=headers
    ) or {}
    account = _fetch_json("https://claude.ai/api/account", headers=headers) or {}
    return {
        "available": True,
        "checked": True,
        "checked_at": _now_iso(),
        "source": "web",
        "cookie_source": cookie_source,
        "organization_id": org_id,
        "windows": windows,
        "overage_limit": overage,
        "account": account,
        "raw_usage": usage,
    }


def fetch_claude_oauth_usage(*, access_token: str) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0",
        "anthropic-beta": "oauth-2025-04-20",
    }
    usage = _fetch_json("https://api.anthropic.com/api/oauth/usage", headers=headers)
    if not usage:
        return {
            "available": False,
            "checked": True,
            "checked_at": _now_iso(),
            "source": "oauth",
            "error": "oauth_usage_unavailable",
        }
    return {
        "available": True,
        "checked": True,
        "checked_at": _now_iso(),
        "source": "oauth",
        "windows": _flatten_windows(usage),
        "raw_usage": usage,
    }


def _extract_next_data(html: str) -> dict[str, Any] | None:
    match = re.search(r"<script[^>]*id=\"__NEXT_DATA__\"[^>]*>(.*?)</script>", html, re.DOTALL)
    if not match:
        return None
    raw = match.group(1).strip()
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, dict) else None


def _search_dict(payload: Any, predicate) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    stack = [payload]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if predicate(item):
                results.append(item)
            for value in item.values():
                stack.append(value)
        elif isinstance(item, list):
            stack.extend(item)
    return results


def _windows_from_next_data(payload: dict[str, Any]) -> list[dict[str, Any]]:
    def looks_like_window(obj: dict[str, Any]) -> bool:
        keys = {str(k).lower() for k in obj.keys()}
        return "percent_used" in keys or "percentused" in keys or "percent_left" in keys or "reset" in keys

    windows: list[dict[str, Any]] = []
    for candidate in _search_dict(payload, looks_like_window):
        window = _window_payload(candidate, "window")
        if window:
            windows.append(window)
    return windows


def fetch_codex_web_usage(*, cookie_header: str, cookie_source: str) -> dict[str, Any]:
    headers = {
        "Cookie": cookie_header,
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/html",
    }
    usage_api = _fetch_json("https://chatgpt.com/backend-api/wham/usage", headers=headers)
    if usage_api:
        return {
            "available": True,
            "checked": True,
            "checked_at": _now_iso(),
            "source": "web",
            "cookie_source": cookie_source,
            "windows": _flatten_windows(usage_api),
            "raw_usage": usage_api,
        }
    html = _fetch_text("https://chatgpt.com/codex/settings/usage", headers=headers)
    if not html:
        return {
            "available": False,
            "checked": True,
            "checked_at": _now_iso(),
            "source": "web",
            "cookie_source": cookie_source,
            "error": "usage_page_unavailable",
        }
    next_data = _extract_next_data(html)
    windows = _windows_from_next_data(next_data) if next_data else []
    return {
        "available": bool(windows),
        "checked": True,
        "checked_at": _now_iso(),
        "source": "web",
        "cookie_source": cookie_source,
        "windows": windows,
        "raw_next_data": next_data or {},
    }


def fetch_codex_oauth_usage(*, access_token: str) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0",
    }
    usage = _fetch_json("https://chatgpt.com/backend-api/wham/usage", headers=headers)
    if not usage:
        return {
            "available": False,
            "checked": True,
            "checked_at": _now_iso(),
            "source": "oauth",
            "error": "oauth_usage_unavailable",
        }
    return {
        "available": True,
        "checked": True,
        "checked_at": _now_iso(),
        "source": "oauth",
        "windows": _flatten_windows(usage),
        "raw_usage": usage,
    }


def load_cookie_for_domain(domain: str, *, env_cookie_header: str | None) -> tuple[str, str] | None:
    domains = [item.strip() for item in domain.split(",") if item.strip()]
    if not domains:
        return None
    if env_cookie_header:
        return env_cookie_header.strip(), "manual"
    headers: list[str] = []
    source = ""
    for item in domains:
        loaded = load_cookie_header(domain=item, env_cookie_header=None)
        if loaded is None:
            continue
        header, source = loaded
        if header:
            headers.append(header)
    if not headers:
        return None
    combined = "; ".join(headers)
    return combined, source or "browser"
