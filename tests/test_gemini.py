from __future__ import annotations

import base64
import json

from gemini import GeminiCliStatsMonitor, GeminiConfig, GeminiQuotaMonitor, GeminiReranker


def _jwt(payload: dict) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8").rstrip("=")
    return f"header.{encoded}.sig"


def test_parse_id_token_claims() -> None:
    email, hosted_domain = GeminiQuotaMonitor._parse_id_token_claims(
        _jwt({"email": "test@example.com", "hd": "example.com"})
    )
    assert email == "test@example.com"
    assert hosted_domain == "example.com"


def test_summarize_quota_buckets_uses_lowest_model_fraction() -> None:
    models, lowest = GeminiQuotaMonitor._summarize_quota_buckets(
        {
            "buckets": [
                {"modelId": "gemini-2.5-pro", "remainingFraction": 0.8, "resetTime": "2026-01-01T00:00:00Z"},
                {"modelId": "gemini-2.5-pro", "remainingFraction": 0.6, "resetTime": "2026-01-02T00:00:00Z"},
                {"modelId": "gemini-2.5-flash", "remainingFraction": 0.9, "resetTime": "2026-01-03T00:00:00Z"},
            ]
        }
    )
    assert len(models) == 2
    assert lowest == 60.0
    pro = next(item for item in models if item["model_id"] == "gemini-2.5-pro")
    assert pro["percent_left"] == 60.0


def test_parse_cli_json_envelope_and_stats_tokens() -> None:
    reranker = GeminiReranker(
        GeminiConfig(
            command="gemini",
            args=[],
            timeout_seconds=5,
            max_output_bytes=1024 * 1024,
            quota_poll_seconds=300,
        )
    )
    payload, stats = reranker._parse_cli_output(
        json.dumps(
            {
                "response": json.dumps({"snippets": [{"id": 1, "score": 0.9, "rationale": "match"}]}),
                "stats": {
                    "models": {
                        "gemini-2.5-pro": {
                            "tokens": {
                                "total": 30,
                                "prompt": 10,
                                "candidates": 20,
                                "input": 10,
                                "cached": 0,
                                "thoughts": 0,
                                "tool": 0,
                            }
                        }
                    }
                },
            }
        )
    )
    assert payload["snippets"][0]["id"] == 1
    assert isinstance(stats, dict)
    tokens = reranker._extract_token_counts_from_stats(stats)
    assert tokens["total"] == 30
    assert tokens["prompt"] == 10
    assert tokens["output"] == 20


def test_parse_cli_output_handles_fenced_response_json() -> None:
    reranker = GeminiReranker(
        GeminiConfig(
            command="gemini",
            args=[],
            timeout_seconds=5,
            max_output_bytes=1024 * 1024,
            quota_poll_seconds=300,
        )
    )
    payload, stats = reranker._parse_cli_output(
        json.dumps(
            {
                "response": "```json\n{\"snippets\":[{\"id\":1,\"score\":0.9,\"rationale\":\"ok\"}]}\n```",
                "stats": {},
            }
        )
    )
    assert payload["snippets"][0]["id"] == 1
    assert stats == {}


def test_parse_cli_output_handles_noise_wrapped_json() -> None:
    reranker = GeminiReranker(
        GeminiConfig(
            command="gemini",
            args=[],
            timeout_seconds=5,
            max_output_bytes=1024 * 1024,
            quota_poll_seconds=300,
        )
    )
    payload, _ = reranker._parse_cli_output(
        "info: warmup complete\n"
        + json.dumps({"snippets": [{"id": 2, "score": 0.8, "rationale": "ok"}]})
    )
    assert payload["snippets"][0]["id"] == 2


def test_quota_status_force_refreshes_synchronously(monkeypatch) -> None:
    monitor = GeminiQuotaMonitor(
        command="gemini",
        timeout_seconds=5,
        poll_interval_seconds=300,
    )
    calls = {"count": 0}

    def fake_refresh() -> None:
        calls["count"] += 1
        monitor._set_state(
            {
                "available": True,
                "checked": True,
                "checked_at": "2026-02-08T00:00:00Z",
                "auth_type": "oauth-personal",
                "account_email": "test@example.com",
                "tier": "",
                "project_id": "",
                "models": [],
                "lowest_percent_left": None,
                "last_error": "",
                "guidance": "ok",
                "next_steps": [],
            }
        )

    monkeypatch.setattr(monitor, "_refresh_once", fake_refresh)
    payload = monitor.status(force=True, allow_poll=True)
    assert calls["count"] == 1
    assert payload["checked"] is True
    assert payload["available"] is True
    assert payload["last_error"] == ""


def test_cli_stats_status_force_refreshes_synchronously(monkeypatch) -> None:
    monitor = GeminiCliStatsMonitor(
        command="gemini",
        args=[],
        timeout_seconds=5,
        poll_interval_seconds=300,
    )
    calls = {"count": 0}

    def fake_refresh() -> None:
        calls["count"] += 1
        monitor._set_state(
            {
                "available": True,
                "checked": True,
                "checked_at": "2026-02-08T00:00:00Z",
                "session_id": "abc",
                "session_model": "gemini-2.5-flash",
                "tool_calls": 1,
                "success_rate_percent": 100.0,
                "models": [],
                "lowest_percent_left": 90.0,
                "last_error": "",
                "guidance": "ok",
                "next_steps": [],
            }
        )

    monkeypatch.setattr(monitor, "_refresh_once", fake_refresh)
    payload = monitor.status(force=True, allow_poll=True)
    assert calls["count"] == 1
    assert payload["checked"] is True
    assert payload["available"] is True
    assert payload["last_error"] == ""


def test_compact_candidates_enforces_prompt_budget_caps() -> None:
    reranker = GeminiReranker(
        GeminiConfig(
            command="gemini",
            args=[],
            timeout_seconds=5,
            max_output_bytes=1024 * 1024,
            quota_poll_seconds=300,
            rerank_max_candidates=4,
            rerank_excerpt_chars=100,
            rerank_prompt_char_budget=450,
        )
    )
    candidates = [
        {
            "id": i,
            "path": f"src/file_{i}.py",
            "start_line": 1,
            "end_line": 80,
            "excerpt": ("x" * 400),
            "base_score": float(10 - i),
            "hit_count": 2,
        }
        for i in range(10)
    ]
    compact = reranker._compact_candidates_for_prompt(candidates, max_results=3)
    assert 1 <= len(compact) <= 4
    assert all(len(item["e"]) <= 140 for item in compact)


def test_plan_retrieval_uses_cache_for_identical_requests(monkeypatch) -> None:
    reranker = GeminiReranker(
        GeminiConfig(
            command="gemini",
            args=[],
            timeout_seconds=5,
            max_output_bytes=1024 * 1024,
            quota_poll_seconds=300,
            planning_cache_seconds=600,
            planning_strategy="gemini",
        )
    )
    reranker._connected = True
    calls = {"count": 0}

    def fake_execute(_prompt: str):
        calls["count"] += 1
        payload = {
            "paths": ["src/a.py"],
            "terms": ["auth", "logic"],
            "rationale": "cached plan test",
        }
        return 0, json.dumps(payload), ""

    monkeypatch.setattr(reranker, "_execute", fake_execute)
    available_paths = ["src/a.py", "src/b.py", "src/c.py"]

    first = reranker.plan_retrieval(
        query="auth logic",
        available_paths=available_paths,
        max_paths=8,
        max_terms=4,
    )
    second = reranker.plan_retrieval(
        query="auth logic",
        available_paths=available_paths,
        max_paths=8,
        max_terms=4,
    )

    assert calls["count"] == 1
    assert first["source"] == "gemini"
    assert second["source"] == "gemini"
    assert second["paths"] == first["paths"]


def test_plan_retrieval_uses_embedded_hints_without_cli(monkeypatch) -> None:
    reranker = GeminiReranker(
        GeminiConfig(
            command="gemini",
            args=[],
            timeout_seconds=5,
            max_output_bytes=1024 * 1024,
            quota_poll_seconds=300,
        )
    )
    reranker._connected = True

    def should_not_execute(_prompt: str):  # pragma: no cover - must not be reached
        raise AssertionError("planner CLI should be skipped for explicit file hints")

    monkeypatch.setattr(reranker, "_execute", should_not_execute)

    plan = reranker.plan_retrieval(
        query="Show SecureLogger.swift logging flow",
        available_paths=[
            "src/core/SecureLogger.swift",
            "src/core/Other.swift",
        ],
        max_paths=5,
        max_terms=5,
        tool_name="context.relevant_code",
        hint_terms=["logging", "flow"],
    )

    assert plan["source"] == "embedded_hints"
    assert "src/core/SecureLogger.swift" in plan["paths"]
    assert "logging" in [term.lower() for term in plan["terms"]]


def test_status_force_does_not_trigger_connect_probe(monkeypatch) -> None:
    reranker = GeminiReranker(
        GeminiConfig(
            command="gemini",
            args=[],
            timeout_seconds=5,
            max_output_bytes=1024 * 1024,
            quota_poll_seconds=300,
        )
    )
    called = {"connect": 0}

    def fake_connect():
        called["connect"] += 1
        return {}

    monkeypatch.setattr(reranker, "connect", fake_connect)
    _ = reranker.status(force=True)
    assert called["connect"] == 0


def test_plan_retrieval_embedded_strategy_skips_cli_without_hints(monkeypatch) -> None:
    reranker = GeminiReranker(
        GeminiConfig(
            command="gemini",
            args=[],
            timeout_seconds=5,
            max_output_bytes=1024 * 1024,
            quota_poll_seconds=300,
            planning_strategy="embedded",
        )
    )
    reranker._connected = True

    def should_not_execute(_prompt: str):  # pragma: no cover - must not be reached
        raise AssertionError("planner CLI should be skipped for embedded strategy")

    monkeypatch.setattr(reranker, "_execute", should_not_execute)
    plan = reranker.plan_retrieval(
        query="find app dependency injection setup",
        available_paths=["src/a.py", "src/b.py"],
        max_paths=4,
        max_terms=4,
        tool_name="context.context_recon",
    )
    assert plan["source"] == "embedded_only"
    assert len(plan["terms"]) >= 1


def test_plan_retrieval_resolves_extensionless_symbol_hint() -> None:
    reranker = GeminiReranker(
        GeminiConfig(
            command="gemini",
            args=[],
            timeout_seconds=5,
            max_output_bytes=1024 * 1024,
            quota_poll_seconds=300,
            planning_strategy="embedded",
        )
    )
    plan = reranker.plan_retrieval(
        query="MultiLaneTimelineView zoom controls gesture",
        available_paths=[
            "AIStageCoach/UI/MultiLaneTimelineView.swift",
            "AIStageCoach/UI/OtherView.swift",
        ],
        max_paths=4,
        max_terms=6,
        tool_name="context.context_recon",
    )
    assert plan["source"] == "embedded_hints"
    assert "AIStageCoach/UI/MultiLaneTimelineView.swift" in plan["paths"]
