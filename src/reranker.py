from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
from typing import Any, Protocol

from cli_reranker import CliRerankerConfig, ClaudeCliReranker, CodexCliReranker
from gemini import GeminiConfig, GeminiReranker


class Reranker(Protocol):
    def status(self, force: bool = False) -> dict[str, Any]: ...

    def rerank(self, query: str, candidates: list[dict], max_results: int) -> list[dict]: ...

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
    ) -> dict[str, Any]: ...


LOG = logging.getLogger(__name__)


def _parse_env_args(value: str) -> list[str]:
    if not value:
        return []
    parsed: Any | None = None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
        return parsed
    if isinstance(parsed, str):
        value = parsed
    try:
        return shlex.split(value)
    except ValueError:
        return [value]


def _normalize_provider(raw: str) -> str:
    value = (raw or "").strip().lower()
    if value in {"auto", "detect"}:
        return "auto"
    if value in {"claude-code", "anthropic"}:
        return "claude"
    if value in {"openai", "codex-cli"}:
        return "codex"
    return value or "gemini"


def _command_exists(command: str) -> bool:
    if not command:
        return False
    return shutil.which(command) is not None


def _resolve_claude_settings(config: dict[str, Any]) -> dict[str, Any]:
    command = str(config.get("claude_command", "claude")).strip() or "claude"
    args = list(config.get("claude_args", []))
    env_command = os.environ.get("CONTEXT_RECON_CLAUDE_COMMAND", "").strip()
    if env_command:
        command = env_command
    env_args = os.environ.get("CONTEXT_RECON_CLAUDE_ARGS", "").strip()
    if env_args:
        args = _parse_env_args(env_args)
    prompt_mode = str(config.get("claude_prompt_mode", "arg")).strip().lower() or "arg"
    env_prompt_mode = os.environ.get("CONTEXT_RECON_CLAUDE_PROMPT_MODE", "").strip().lower()
    if env_prompt_mode:
        prompt_mode = env_prompt_mode
    usage_source = str(config.get("claude_usage_source", "auto")).strip().lower() or "auto"
    env_usage_source = os.environ.get("CONTEXT_RECON_CLAUDE_USAGE_SOURCE", "").strip().lower()
    if env_usage_source:
        usage_source = env_usage_source
    cookie_source = str(config.get("claude_cookie_source", "auto")).strip().lower() or "auto"
    env_cookie_source = os.environ.get("CONTEXT_RECON_CLAUDE_COOKIE_SOURCE", "").strip().lower()
    if env_cookie_source:
        cookie_source = env_cookie_source
    cookie_header = str(config.get("claude_cookie_header", "") or "").strip()
    env_cookie_header = os.environ.get("CONTEXT_RECON_CLAUDE_COOKIE_HEADER", "").strip()
    if env_cookie_header:
        cookie_header = env_cookie_header
    return {
        "command": command,
        "args": args,
        "prompt_mode": prompt_mode,
        "usage_source": usage_source,
        "cookie_source": cookie_source,
        "cookie_header": cookie_header,
    }


def _resolve_codex_settings(config: dict[str, Any]) -> dict[str, Any]:
    command = str(config.get("codex_command", "codex")).strip() or "codex"
    args = list(config.get("codex_args", []))
    env_command = os.environ.get("CONTEXT_RECON_CODEX_COMMAND", "").strip()
    if env_command:
        command = env_command
    env_args = os.environ.get("CONTEXT_RECON_CODEX_ARGS", "").strip()
    if env_args:
        args = _parse_env_args(env_args)
    prompt_mode = str(config.get("codex_prompt_mode", "arg")).strip().lower() or "arg"
    env_prompt_mode = os.environ.get("CONTEXT_RECON_CODEX_PROMPT_MODE", "").strip().lower()
    if env_prompt_mode:
        prompt_mode = env_prompt_mode
    usage_source = str(config.get("codex_usage_source", "auto")).strip().lower() or "auto"
    env_usage_source = os.environ.get("CONTEXT_RECON_CODEX_USAGE_SOURCE", "").strip().lower()
    if env_usage_source:
        usage_source = env_usage_source
    cookie_source = str(config.get("codex_cookie_source", "auto")).strip().lower() or "auto"
    env_cookie_source = os.environ.get("CONTEXT_RECON_CODEX_COOKIE_SOURCE", "").strip().lower()
    if env_cookie_source:
        cookie_source = env_cookie_source
    cookie_header = str(config.get("codex_cookie_header", "") or "").strip()
    env_cookie_header = os.environ.get("CONTEXT_RECON_CODEX_COOKIE_HEADER", "").strip()
    if env_cookie_header:
        cookie_header = env_cookie_header
    return {
        "command": command,
        "args": args,
        "prompt_mode": prompt_mode,
        "usage_source": usage_source,
        "cookie_source": cookie_source,
        "cookie_header": cookie_header,
    }


def _resolve_gemini_settings(config: dict[str, Any]) -> dict[str, Any]:
    command = str(config.get("gemini_command", "gemini")).strip() or "gemini"
    args = list(config.get("gemini_args", []))
    env_command = os.environ.get("CONTEXT_RECON_GEMINI_COMMAND", "").strip()
    if env_command:
        command = env_command
    env_args = os.environ.get("CONTEXT_RECON_GEMINI_ARGS", "").strip()
    if env_args:
        args = _parse_env_args(env_args)
    return {"command": command, "args": args}


def _select_provider(
    *,
    preferred: str,
    order: list[str],
    commands: dict[str, str],
) -> str:
    if preferred == "auto":
        for candidate in order:
            if _command_exists(commands.get(candidate, "")):
                return candidate
        return order[0]
    if _command_exists(commands.get(preferred, "")):
        return preferred
    for candidate in order:
        if candidate == preferred:
            continue
        if _command_exists(commands.get(candidate, "")):
            LOG.warning(
                "Preferred CLI '%s' not found; falling back to '%s'.", preferred, candidate
            )
            return candidate
    return preferred


def build_reranker(
    *,
    config: dict[str, Any],
    timeout_seconds: int,
    max_output_bytes: int,
) -> Reranker:
    provider = _normalize_provider(str(config.get("reranker_provider", "auto")))
    env_provider_raw = os.environ.get("CONTEXT_RECON_RERANK_PROVIDER", "").strip()
    if env_provider_raw:
        provider = _normalize_provider(env_provider_raw)

    claude_settings = _resolve_claude_settings(config)
    codex_settings = _resolve_codex_settings(config)
    gemini_settings = _resolve_gemini_settings(config)
    order = ["gemini", "claude", "codex"]
    commands = {
        "gemini": str(gemini_settings["command"]),
        "claude": str(claude_settings["command"]),
        "codex": str(codex_settings["command"]),
    }
    provider = _select_provider(preferred=provider, order=order, commands=commands)

    if provider == "claude":
        command = str(claude_settings["command"])
        args = list(claude_settings["args"])
        prompt_mode = str(claude_settings["prompt_mode"])
        usage_source = str(claude_settings["usage_source"])
        cookie_source = str(claude_settings["cookie_source"])
        cookie_header = str(claude_settings["cookie_header"])
        return ClaudeCliReranker(
            CliRerankerConfig(
                provider="claude",
                command=command,
                args=args,
                prompt_mode=prompt_mode,
                usage_source=usage_source,
                cookie_source=cookie_source,
                cookie_header=cookie_header,
                cookie_domain="claude.ai",
                timeout_seconds=timeout_seconds,
                max_output_bytes=max_output_bytes,
                planning_max_catalog_paths=int(config.get("gemini_planning_max_catalog_paths", 48)),
                planning_cache_seconds=int(config.get("gemini_planning_cache_seconds", 600)),
                planning_strategy=str(config.get("gemini_planning_strategy", "embedded")),
                rerank_max_candidates=int(config.get("gemini_rerank_max_candidates", 8)),
                rerank_excerpt_chars=int(config.get("gemini_rerank_excerpt_chars", 280)),
                rerank_prompt_char_budget=int(config.get("gemini_rerank_prompt_char_budget", 9000)),
            )
        )

    if provider == "codex":
        command = str(codex_settings["command"])
        args = list(codex_settings["args"])
        prompt_mode = str(codex_settings["prompt_mode"])
        usage_source = str(codex_settings["usage_source"])
        cookie_source = str(codex_settings["cookie_source"])
        cookie_header = str(codex_settings["cookie_header"])
        return CodexCliReranker(
            CliRerankerConfig(
                provider="codex",
                command=command,
                args=args,
                prompt_mode=prompt_mode,
                usage_source=usage_source,
                cookie_source=cookie_source,
                cookie_header=cookie_header,
                cookie_domain="chatgpt.com,openai.com",
                timeout_seconds=timeout_seconds,
                max_output_bytes=max_output_bytes,
                planning_max_catalog_paths=int(config.get("gemini_planning_max_catalog_paths", 48)),
                planning_cache_seconds=int(config.get("gemini_planning_cache_seconds", 600)),
                planning_strategy=str(config.get("gemini_planning_strategy", "embedded")),
                rerank_max_candidates=int(config.get("gemini_rerank_max_candidates", 8)),
                rerank_excerpt_chars=int(config.get("gemini_rerank_excerpt_chars", 280)),
                rerank_prompt_char_budget=int(config.get("gemini_rerank_prompt_char_budget", 9000)),
            )
        )

    command = str(gemini_settings["command"])
    args = list(gemini_settings["args"])
    return GeminiReranker(
        GeminiConfig(
            command=command,
            args=args,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
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
