from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.json_schema import validate_gemini_rerank_payload
from utils.rationale import build_factual_plan_rationale, build_factual_rerank_rationale
from provider_usage import (
    fetch_claude_oauth_usage,
    fetch_claude_web_usage,
    fetch_codex_oauth_usage,
    fetch_codex_web_usage,
    load_claude_oauth_token,
    load_codex_oauth_token,
    load_cookie_for_domain,
)


@dataclass(frozen=True)
class CliRerankerConfig:
    provider: str
    command: str
    args: list[str]
    prompt_mode: str
    usage_source: str
    cookie_source: str
    cookie_header: str
    cookie_domain: str
    timeout_seconds: int
    max_output_bytes: int
    planning_max_catalog_paths: int
    planning_cache_seconds: int
    planning_strategy: str
    rerank_max_candidates: int
    rerank_excerpt_chars: int
    rerank_prompt_char_budget: int


class BaseCliReranker:
    def __init__(self, config: CliRerankerConfig) -> None:
        self.config = config
        self._connected = False
        self._connection_checked = False
        self._last_error = "not_checked"
        self._last_error_detail = ""
        self._usage: dict[str, Any] = {
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
            "tokens_source": "",
        }
        self._plan_cache_lock = threading.Lock()
        self._plan_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._external_usage: dict[str, Any] = {
            "available": False,
            "checked": False,
            "checked_at": "",
            "source": "",
            "error": "not_checked",
        }
        self._external_usage_checked_at = 0.0
        self._external_usage_ttl_seconds = 120

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _build_command(self, *, prompt: str, schema_json: str) -> tuple[list[str], list[Path], Path | None]:
        raise NotImplementedError

    def _run_process(self, command: list[str], *, prompt: str | None) -> tuple[int, str, str] | None:
        self._usage["cli_invocations_total"] += 1
        try:
            proc = subprocess.Popen(
                command,
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

    def _execute(self, prompt: str, schema: dict[str, Any]) -> tuple[int, str, str] | None:
        schema_json = json.dumps(schema, ensure_ascii=True)
        command, cleanup_paths, output_path = self._build_command(prompt=prompt, schema_json=schema_json)
        try:
            result = self._run_process(
                command,
                prompt=prompt if self.config.prompt_mode == "stdin" else None,
            )
            if result is None:
                return None
            returncode, stdout, stderr = result
            if output_path is not None:
                if not output_path.exists():
                    self._connected = False
                    self._last_error = "invalid_json_from_cli"
                    self._last_error_detail = "output_file_missing"
                    self._connection_checked = True
                    return None
                data = output_path.read_bytes()
                if len(data) > self.config.max_output_bytes:
                    self._connected = False
                    self._last_error = "output_limit_exceeded"
                    self._last_error_detail = ""
                    self._connection_checked = True
                    return None
                stdout = data.decode("utf-8", errors="replace")
            return returncode, stdout, stderr
        finally:
            for path in cleanup_paths:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _decode_json_payload(self, raw: str) -> Any:
        text = (raw or "").strip()
        if not text:
            raise ValueError("invalid_json_from_cli")

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
        for block in fenced_blocks:
            block_text = block.strip()
            if not block_text:
                continue
            try:
                return json.loads(block_text)
            except json.JSONDecodeError:
                continue

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

    def _parse_cli_output(self, stdout: str) -> dict[str, Any]:
        decoded = self._decode_json_payload(stdout)
        if isinstance(decoded, dict):
            return decoded
        raise ValueError("invalid_json_from_cli")

    def _estimate_tokens(self, text: str) -> int:
        cleaned = (text or "").strip()
        if not cleaned:
            return 0
        return max(1, int(round(len(cleaned) / 4)))

    def _coerce_int(self, value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _extract_usage_from_dict(self, payload: dict[str, Any]) -> dict[str, int] | None:
        def pick(keys: list[str]) -> int | None:
            for key in keys:
                if key in payload:
                    value = self._coerce_int(payload.get(key))
                    if value is not None:
                        return value
            return None

        input_tokens = pick(["input_tokens", "prompt_tokens", "inputTokens", "promptTokens", "input"])
        output_tokens = pick(["output_tokens", "completion_tokens", "outputTokens", "completionTokens", "output"])
        total_tokens = pick(["total_tokens", "totalTokens", "total"])
        if input_tokens is None and output_tokens is None and total_tokens is None:
            return None
        if total_tokens is None:
            total_tokens = (input_tokens or 0) + (output_tokens or 0)
        return {
            "input": input_tokens or 0,
            "output": output_tokens or 0,
            "total": total_tokens or 0,
        }

    def _extract_usage(self, stdout: str, payload: dict[str, Any]) -> dict[str, int] | None:
        try:
            decoded = self._decode_json_payload(stdout)
        except ValueError:
            decoded = None
        candidates: list[dict[str, Any]] = []
        if isinstance(decoded, dict):
            candidates.append(decoded)
            for key in ("usage", "token_usage", "tokenUsage", "modelUsage", "stats"):
                nested = decoded.get(key)
                if isinstance(nested, dict):
                    candidates.append(nested)
        for key in ("usage", "token_usage", "tokenUsage", "modelUsage", "stats"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                candidates.append(nested)
        candidates.append(payload)
        for candidate in candidates:
            usage = self._extract_usage_from_dict(candidate)
            if usage:
                return usage
        return None

    def _update_usage_from_tokens(self, *, input_tokens: int, output_tokens: int, total_tokens: int, source: str) -> None:
        self._usage["gemini_total_tokens"] += int(total_tokens)
        self._usage["gemini_prompt_tokens"] += int(input_tokens)
        self._usage["gemini_output_tokens"] += int(output_tokens)
        self._usage["gemini_input_tokens"] += int(input_tokens)
        self._usage["last_call_tokens"] = {
            "total": int(total_tokens),
            "prompt": int(input_tokens),
            "output": int(output_tokens),
            "input": int(input_tokens),
            "cached": 0,
            "thought": 0,
            "tool": 0,
        }
        self._usage["last_usage_update_at"] = self._now_iso()
        self._usage["tokens_source"] = source

    def _apply_usage(self, *, prompt: str, stdout: str, payload: dict[str, Any]) -> None:
        usage = self._extract_usage(stdout, payload)
        if usage is not None:
            self._update_usage_from_tokens(
                input_tokens=usage.get("input", 0),
                output_tokens=usage.get("output", 0),
                total_tokens=usage.get("total", 0),
                source="cli_usage",
            )
            return
        estimated_input = self._estimate_tokens(prompt)
        payload_text = json.dumps(payload, ensure_ascii=True)
        estimated_output = self._estimate_tokens(payload_text)
        self._update_usage_from_tokens(
            input_tokens=estimated_input,
            output_tokens=estimated_output,
            total_tokens=estimated_input + estimated_output,
            source="estimated",
        )

    def _build_prompt(self, query: str, candidates: list[dict], max_results: int) -> str:
        serialized = json.dumps(self._compact_candidates_for_prompt(candidates, max_results=max_results), ensure_ascii=True)
        query_text = self._compact_query_text(query, max_terms=14, max_chars=180)
        return (
            "Rank local code excerpts for relevance.\n"
            "Return JSON only using: {\"snippets\": [{\"id\": int, \"score\": float, \"rationale\": string}]}.\n"
            "Use only provided candidates. Do not invent code or files.\n"
            "Keep rationale concise, factual, and grounded only in provided query/candidate text.\n"
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

        cleaned = [line.rstrip() for line in lines]
        while cleaned and not cleaned[0].strip():
            cleaned.pop(0)
        while cleaned and not cleaned[-1].strip():
            cleaned.pop()

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
            "Keep rationale concise, factual, and based only on provided hints/query/catalog.\n"
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
        provider_label = self.config.provider.title()

        if code == "not_checked":
            return (
                f"{provider_label} CLI connection has not been checked yet.",
                ["Run context.relevant_code or context.index_inspection to trigger a connection check."],
            )
        if code == "cli_not_found_or_not_executable":
            return (
                f"{provider_label} CLI was not found or cannot be executed.",
                [
                    f"Run `{command_hint} --help` in this same environment to verify the CLI is installed and on PATH.",
                    "If your executable name is different, set the command in config.yaml or env.",
                    "After fixing the command, run another context request; restart is not required.",
                ],
            )
        if code == "timeout":
            return (
                f"{provider_label} CLI timed out before returning a response.",
                [
                    "Run the CLI directly to ensure it is not waiting for interactive login.",
                    "Authenticate the CLI in this same shell environment if prompted.",
                    "Increase `gemini_timeout_seconds` in `config.yaml` if the local model call is slow.",
                    "After fixing authentication/timeout, run another context request; restart is not required.",
                ],
            )
        if code == "output_limit_exceeded":
            return (
                f"{provider_label} CLI output exceeded the configured size limit.",
                [
                    "Reduce request size (`max_results` or `max_excerpt_lines`) for context.relevant_code.",
                    "Optionally increase `gemini_max_output_bytes` in `config.yaml`.",
                    "Run another context request after adjustment; restart is not required.",
                ],
            )
        if code == "invalid_json_from_cli":
            return (
                f"{provider_label} CLI returned a non-JSON or invalid JSON response.",
                [
                    "Ensure your CLI invocation returns raw JSON to stdout.",
                    "Confirm schema/print-mode flags are set for structured output.",
                    "Run another context request after adjustment; restart is not required.",
                ],
            )
        if code == "auth_interactive_required":
            return (
                f"{provider_label} CLI requires sign-in in this environment before non-interactive JSON calls can run.",
                [
                    "Run the CLI once in a regular terminal and complete the auth flow.",
                    "After successful sign-in, re-run context.index_inspection.",
                ],
            )
        if code.startswith("cli_error"):
            return (
                f"{provider_label} CLI returned an execution error.",
                [
                    "Run the same CLI command in this environment to see the full error.",
                    "If the error indicates authentication, sign in to the CLI in this shell environment.",
                    "After fixing the CLI error, run another context request; restart is not required.",
                ],
            )
        return (
            f"{provider_label} CLI is unavailable; deterministic fallback ranking is active.",
            ["Resolve CLI availability/authentication issues and retry a context request."],
        )

    def status(self, force: bool = False) -> dict[str, Any]:
        external_usage = self._refresh_external_usage(force=force)
        guidance, steps = self._guidance() if not self._connected else (f"{self.config.provider.title()} CLI connected.", [])
        return {
            "provider": self.config.provider,
            "command": self.config.command,
            "args": list(self.config.args),
            "connected": self._connected,
            "checked": self._connection_checked,
            "last_error": self._last_error,
            "error_detail": self._last_error_detail,
            "guidance": guidance,
            "next_steps": steps,
            "fallback_active": not self._connected,
            "usage": dict(self._usage),
            "usage_notes": "Token values use CLI usage fields when available; otherwise estimated from prompt/output size.",
            "quota": {"available": False, "checked": False, "last_error": "not_supported", "guidance": ""},
            "cli_stats": {"available": False, "checked": False, "last_error": "not_supported", "guidance": ""},
            "external_usage": external_usage,
        }

    def _refresh_external_usage(self, *, force: bool) -> dict[str, Any]:
        if self.config.provider not in {"claude", "codex"}:
            return dict(self._external_usage)
        now = time.monotonic()
        if not force and (now - self._external_usage_checked_at) < self._external_usage_ttl_seconds:
            return dict(self._external_usage)
        usage_source = (self.config.usage_source or "web").strip().lower()
        if usage_source == "off":
            self._external_usage = {
                "available": False,
                "checked": True,
                "checked_at": self._now_iso(),
                "source": "off",
                "error": "disabled",
            }
            self._external_usage_checked_at = now
            return dict(self._external_usage)
        if usage_source in {"oauth", "auto"}:
            if self.config.provider == "claude":
                token = load_claude_oauth_token()
                if token:
                    result = fetch_claude_oauth_usage(access_token=token)
                    if result.get("available") or usage_source == "oauth":
                        self._external_usage = result
                        self._external_usage_checked_at = now
                        return dict(self._external_usage)
            else:
                token = load_codex_oauth_token()
                if token:
                    result = fetch_codex_oauth_usage(access_token=token)
                    if result.get("available") or usage_source == "oauth":
                        self._external_usage = result
                        self._external_usage_checked_at = now
                        return dict(self._external_usage)
            if usage_source == "oauth":
                self._external_usage = {
                    "available": False,
                    "checked": True,
                    "checked_at": self._now_iso(),
                    "source": "oauth",
                    "error": "oauth_unavailable",
                }
                self._external_usage_checked_at = now
                return dict(self._external_usage)
        cookie_header = self.config.cookie_header.strip() if self.config.cookie_header else ""
        cookie_source = ""
        if self.config.cookie_source != "off":
            loaded = load_cookie_for_domain(
                domain=self.config.cookie_domain, env_cookie_header=cookie_header or None
            )
            if loaded is not None:
                cookie_header, cookie_source = loaded
        if not cookie_header:
            self._external_usage = {
                "available": False,
                "checked": True,
                "checked_at": self._now_iso(),
                "source": "web",
                "error": "cookies_unavailable",
            }
            self._external_usage_checked_at = now
            return dict(self._external_usage)
        if self.config.provider == "claude":
            self._external_usage = fetch_claude_web_usage(
                cookie_header=cookie_header,
                cookie_source=cookie_source or "browser",
            )
        else:
            self._external_usage = fetch_codex_web_usage(
                cookie_header=cookie_header,
                cookie_source=cookie_source or "browser",
            )
        self._external_usage_checked_at = now
        return dict(self._external_usage)

    def connect(self) -> dict[str, Any]:
        prompt = self._build_prompt(
            "connection probe",
            [{"id": 0, "path": "probe", "start_line": 1, "end_line": 1, "excerpt": "probe", "base_score": 1.0}],
            1,
        )
        result = self._execute(prompt, schema=_rerank_schema())
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
            payload = self._parse_cli_output(stdout)
            validate_gemini_rerank_payload(payload)
        except (json.JSONDecodeError, ValueError):
            self._connected = False
            lower = stdout.lower()
            if "authorize" in lower or "auth" in lower or "login" in lower:
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
                "rationale": "Embedded planning strategy active; planner call skipped.",
                "source": "embedded_only",
            }

        if not self._connected:
            self.connect()
        if not self._connected:
            self._usage["fallback_total"] += 1
            return {
                "paths": [],
                "terms": fallback_terms,
                "rationale": "CLI unavailable; deterministic query-term fallback.",
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
        result = self._execute(prompt, schema=_plan_schema())
        if result is None:
            self._usage["fallback_total"] += 1
            return {
                "paths": [],
                "terms": fallback_terms,
                "rationale": "CLI execution failed; deterministic query-term fallback.",
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
                "rationale": "CLI execution error; deterministic query-term fallback.",
                "source": "deterministic_fallback",
            }

        try:
            payload = self._parse_cli_output(stdout)
            raw_paths = payload.get("paths", [])
            raw_terms = payload.get("terms", [])
            if not isinstance(raw_paths, list) or not isinstance(raw_terms, list):
                raise ValueError("invalid_json_from_cli")
            self._apply_usage(prompt=prompt, stdout=stdout, payload=payload)
        except (json.JSONDecodeError, ValueError, TypeError):
            self._connected = False
            self._last_error = "invalid_json_from_cli"
            self._last_error_detail = ""
            self._usage["fallback_total"] += 1
            return {
                "paths": [],
                "terms": fallback_terms,
                "rationale": "CLI returned invalid plan JSON; deterministic query-term fallback.",
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

        plan = {
            "paths": selected_paths,
            "terms": selected_terms,
            "rationale": build_factual_plan_rationale(
                selected_paths=selected_paths,
                selected_terms=selected_terms,
            ),
            "source": self.config.provider,
        }
        with self._plan_cache_lock:
            self._plan_cache[cache_key] = (time.monotonic(), dict(plan))
            if len(self._plan_cache) > 200:
                oldest_key = min(self._plan_cache.items(), key=lambda item: item[1][0])[0]
                self._plan_cache.pop(oldest_key, None)
        return plan

    def rerank(self, query: str, candidates: list[dict], max_results: int) -> list[dict]:
        if not candidates:
            return []

        self._usage["requests_total"] += 1
        self._usage["last_request_at"] = self._now_iso()

        if not self._connected:
            self.connect()
        if not self._connected:
            self._usage["fallback_total"] += 1
            return self._deterministic_fallback(candidates, max_results)

        prompt = self._build_prompt(query, candidates, max_results)
        result = self._execute(prompt, schema=_rerank_schema())
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
            payload = self._parse_cli_output(stdout)
            normalized = validate_gemini_rerank_payload(payload)
            self._apply_usage(prompt=prompt, stdout=stdout, payload=payload)
        except (json.JSONDecodeError, ValueError):
            self._connected = False
            lower = stdout.lower()
            if "authorize" in lower or "auth" in lower or "login" in lower:
                self._last_error = "auth_interactive_required"
                self._last_error_detail = stdout[:1000]
            else:
                self._last_error = "invalid_json_from_cli"
                self._last_error_detail = stdout[:1000]
            self._usage["fallback_total"] += 1
            return self._deterministic_fallback(candidates, max_results)

        candidate_map = {int(item["id"]): item for item in candidates}
        candidate_ids = set(candidate_map.keys())
        seen: set[int] = set()
        filtered: list[dict] = []
        for item in normalized:
            cid = item["id"]
            if cid not in candidate_ids or cid in seen:
                continue
            candidate = candidate_map.get(cid)
            if candidate is None:
                continue
            seen.add(cid)
            filtered.append(
                {
                    "id": cid,
                    "score": float(item["score"]),
                    "rationale": build_factual_rerank_rationale(
                        query=query,
                        excerpt=str(candidate.get("excerpt", "")),
                        path=str(candidate.get("path", "")),
                    ),
                }
            )
            if len(filtered) >= max_results:
                break

        if not filtered:
            self._usage["fallback_total"] += 1
            return self._deterministic_fallback(candidates, max_results)

        self._usage["cli_success_total"] += 1
        self._usage["last_success_at"] = self._now_iso()
        return filtered


class ClaudeCliReranker(BaseCliReranker):
    def _build_command(self, *, prompt: str, schema_json: str) -> tuple[list[str], list[Path], Path | None]:
        cmd = [self.config.command, *self.config.args]
        has_print = any(arg in {"-p", "--print"} for arg in cmd)
        if not has_print:
            cmd.append("-p")
        if self.config.prompt_mode == "arg":
            insert_at = None
            for idx, arg in enumerate(cmd):
                if arg in {"-p", "--print"}:
                    insert_at = idx + 1
            if insert_at is None:
                cmd.append(prompt)
            else:
                cmd.insert(insert_at, prompt)
        if not any(arg == "--output-format" or arg.startswith("--output-format=") for arg in cmd):
            cmd.extend(["--output-format", "json"])
        if not any(arg == "--json-schema" or arg.startswith("--json-schema=") for arg in cmd):
            cmd.extend(["--json-schema", schema_json])
        return cmd, [], None

    def _parse_cli_output(self, stdout: str) -> dict[str, Any]:
        decoded = self._decode_json_payload(stdout)
        if isinstance(decoded, dict) and isinstance(decoded.get("result"), str):
            return super()._parse_cli_output(decoded["result"])
        return super()._parse_cli_output(stdout)


class CodexCliReranker(BaseCliReranker):
    def _build_command(self, *, prompt: str, schema_json: str) -> tuple[list[str], list[Path], Path | None]:
        cmd = [self.config.command, "exec", *self.config.args]
        if not _has_flag(cmd, "--sandbox"):
            cmd.extend(["--sandbox", "read-only"])
        if not _has_flag(cmd, "--ask-for-approval"):
            cmd.extend(["--ask-for-approval", "never"])

        schema_file = tempfile.NamedTemporaryFile(prefix="context_recon_schema_", suffix=".json", delete=False)
        schema_path = Path(schema_file.name)
        schema_file.write(schema_json.encode("utf-8"))
        schema_file.close()

        output_file = tempfile.NamedTemporaryFile(prefix="context_recon_output_", suffix=".json", delete=False)
        output_path = Path(output_file.name)
        output_file.close()

        cmd.extend(["--output-schema", str(schema_path)])
        cmd.extend(["--output-last-message", str(output_path)])
        if self.config.prompt_mode == "arg":
            cmd.extend(["--", prompt])
        return cmd, [schema_path, output_path], output_path


def _has_flag(args: list[str], flag: str) -> bool:
    if flag in args:
        return True
    prefix = flag + "="
    return any(arg.startswith(prefix) for arg in args)


def _rerank_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["snippets"],
        "properties": {
            "snippets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "score", "rationale"],
                    "properties": {
                        "id": {"type": "integer"},
                        "score": {"type": "number"},
                        "rationale": {"type": "string"},
                    },
                },
            }
        },
    }


def _plan_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["paths", "terms", "rationale"],
        "properties": {
            "paths": {"type": "array", "items": {"type": "string"}},
            "terms": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"},
        },
    }
