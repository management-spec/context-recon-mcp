from __future__ import annotations

from types import SimpleNamespace

from server import ContextEngine, cleanup_managed_mcp_servers, cleanup_orphaned_mcp_servers, update_managed_mcp_repository


def _engine_stub() -> ContextEngine:
    return object.__new__(ContextEngine)


def test_normalize_scope_paths_accepts_csv_string() -> None:
    engine = _engine_stub()
    result = engine._normalize_scope_paths("src, tests ,docs/api")
    assert result == ["src", "tests", "docs/api"]


def test_normalize_scope_paths_accepts_list_and_strips_values() -> None:
    engine = _engine_stub()
    result = engine._normalize_scope_paths([" src ", "", "tests", "  "])
    assert result == ["src", "tests"]


def test_normalize_scope_paths_rejects_invalid_shapes() -> None:
    engine = _engine_stub()
    assert engine._normalize_scope_paths(None) is None
    assert engine._normalize_scope_paths("") is None
    assert engine._normalize_scope_paths([]) is None
    assert engine._normalize_scope_paths(123) is None


def test_cleanup_managed_mcp_servers_kills_matching_non_current_pids(monkeypatch) -> None:
    ps_output = "\n".join(
        [
            "PID PPID COMMAND",
            "101 1 /usr/bin/python /tmp/context-recon-mcp/src/server.py",
            "102 77 /usr/bin/python /tmp/context-recon-mcp/src/server.py",
            "103 1 /usr/bin/python /tmp/gemini-context-engine-mcp/src/server.py",
            "104 1 /usr/bin/python /tmp/other_server.py",
        ]
    )

    monkeypatch.setattr(
        "server.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=ps_output),
    )
    killed: list[int] = []
    monkeypatch.setattr("server.os.kill", lambda pid, sig: killed.append(pid))

    result = cleanup_managed_mcp_servers(current_pid=102, orphan_only=False)

    assert result["matched"] == 3
    assert result["killed"] == 2
    assert sorted(result["killed_pids"]) == [101, 103]
    assert sorted(killed) == [101, 103]
    assert result["skipped_current"] is True


def test_cleanup_orphaned_mcp_servers_filters_non_orphans(monkeypatch) -> None:
    ps_output = "\n".join(
        [
            "PID PPID COMMAND",
            "201 1 /usr/bin/python /tmp/context-recon-mcp/src/server.py",
            "202 88 /usr/bin/python /tmp/context-recon-mcp/src/server.py",
            "203 1 /usr/bin/python /tmp/gemini-context-engine-mcp/src/server.py",
        ]
    )

    monkeypatch.setattr(
        "server.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=ps_output),
    )
    killed: list[int] = []
    monkeypatch.setattr("server.os.kill", lambda pid, sig: killed.append(pid))

    killed_count = cleanup_orphaned_mcp_servers(current_pid=203)

    assert killed_count == 1
    assert killed == [201]


def test_update_managed_mcp_repository_pulls_when_clean(monkeypatch, tmp_path) -> None:
    head_values = iter(
        [
            "1111111111111111111111111111111111111111\n",
            "2222222222222222222222222222222222222222\n",
        ]
    )
    seen_commands: list[list[str]] = []

    def fake_run(command, *args, **kwargs):
        cmd = list(command)
        seen_commands.append(cmd)
        if cmd[:2] == ["git", "--version"]:
            return SimpleNamespace(returncode=0, stdout="git version 2.42.0\n", stderr="")
        if cmd[:3] == ["git", "rev-parse", "--is-inside-work-tree"]:
            return SimpleNamespace(returncode=0, stdout="true\n", stderr="")
        if cmd[:3] == ["git", "status", "--porcelain"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd[:3] == ["git", "rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout=next(head_values), stderr="")
        if cmd[:3] == ["git", "pull", "--ff-only"]:
            return SimpleNamespace(returncode=0, stdout="Updating 1111111..2222222\n", stderr="")
        if cmd[:3] == ["git", "diff", "--name-only"]:
            return SimpleNamespace(returncode=0, stdout="src/server.py\nREADME.md\n", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr("server.subprocess.run", fake_run)

    result = update_managed_mcp_repository(
        repo_root=tmp_path,
        remote="origin",
        branch="main",
        allow_dirty=False,
        timeout_seconds=45,
    )

    assert result["ok"] is True
    assert result["updated"] is True
    assert result["branch"] == "main"
    assert result["changed_files_count"] == 2
    assert result["changed_files"] == ["src/server.py", "README.md"]
    assert any(cmd[:3] == ["git", "pull", "--ff-only"] for cmd in seen_commands)


def test_update_managed_mcp_repository_skips_dirty_worktree(monkeypatch, tmp_path) -> None:
    seen_commands: list[list[str]] = []

    def fake_run(command, *args, **kwargs):
        cmd = list(command)
        seen_commands.append(cmd)
        if cmd[:2] == ["git", "--version"]:
            return SimpleNamespace(returncode=0, stdout="git version 2.42.0\n", stderr="")
        if cmd[:3] == ["git", "rev-parse", "--is-inside-work-tree"]:
            return SimpleNamespace(returncode=0, stdout="true\n", stderr="")
        if cmd[:3] == ["git", "status", "--porcelain"]:
            return SimpleNamespace(returncode=0, stdout=" M src/server.py\n", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr("server.subprocess.run", fake_run)

    result = update_managed_mcp_repository(
        repo_root=tmp_path,
        remote="origin",
        branch="main",
        allow_dirty=False,
        timeout_seconds=45,
    )

    assert result["ok"] is False
    assert result["skipped"] is True
    assert result["error"] == "dirty_worktree"
    assert all(cmd[:3] != ["git", "pull", "--ff-only"] for cmd in seen_commands)
