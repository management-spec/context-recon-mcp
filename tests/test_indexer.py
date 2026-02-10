from __future__ import annotations

from pathlib import Path

from indexer import ContextIndexer


def test_scan_all_respects_gitignore_and_size_limit(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (tmp_path / "keep.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "ignored.py").write_text("print('no')\n", encoding="utf-8")

    indexer = ContextIndexer(
        workspace_root=tmp_path,
        roots=["."],
        db_path=".context_engine/index.db",
        max_file_bytes=1024,
        ignore_patterns=[],
    )

    indexed, ignored = indexer.scan_all()

    assert indexed >= 1
    assert ignored >= 1
    paths = list(indexer.existing_paths())
    assert "keep.py" in paths
    assert "ignored.py" not in paths


def test_incremental_index_update(tmp_path: Path) -> None:
    file_path = tmp_path / "src.py"
    file_path.write_text("x = 1\n", encoding="utf-8")

    indexer = ContextIndexer(
        workspace_root=tmp_path,
        roots=["."],
        db_path=".context_engine/index.db",
        max_file_bytes=1024,
        ignore_patterns=[],
    )
    indexer.scan_all()

    record = indexer.get_record("src.py")
    assert record is not None

    file_path.write_text("x = 2\n", encoding="utf-8")
    assert indexer.try_index(file_path) is True

    updated = indexer.get_record("src.py")
    assert updated is not None
    assert updated.mtime >= record.mtime


def test_hidden_paths_are_ignored_by_default(tmp_path: Path) -> None:
    (tmp_path / ".hidden.py").write_text("print('no')\n", encoding="utf-8")
    hidden_dir = tmp_path / ".hidden_dir"
    hidden_dir.mkdir(parents=True, exist_ok=True)
    (hidden_dir / "inside.py").write_text("print('no')\n", encoding="utf-8")
    (tmp_path / "visible.py").write_text("print('yes')\n", encoding="utf-8")

    indexer = ContextIndexer(
        workspace_root=tmp_path,
        roots=["."],
        db_path=".context_engine/index.db",
        max_file_bytes=1024,
        ignore_patterns=[],
    )

    indexer.scan_all()
    paths = list(indexer.existing_paths())
    assert "visible.py" in paths
    assert ".hidden.py" not in paths
    assert ".hidden_dir/inside.py" not in paths


def test_query_usage_is_recorded_and_ranked(tmp_path: Path) -> None:
    (tmp_path / "visible.py").write_text("print('yes')\n", encoding="utf-8")
    indexer = ContextIndexer(
        workspace_root=tmp_path,
        roots=["."],
        db_path=".context_engine/index.db",
        max_file_bytes=1024,
        ignore_patterns=[],
    )
    indexer.scan_all()
    indexer.record_query_usage(tool="context.context_recon", query="secure logger")
    indexer.record_query_usage(tool="context.context_recon", query="secure logger")
    indexer.record_query_usage(tool="context.code_search", query="MainActor")

    top = indexer.top_queries(limit=5)
    assert len(top) >= 2
    assert top[0]["query"] == "secure logger"
    assert int(top[0]["count"]) == 2


def test_scan_all_purges_stale_hidden_rows(tmp_path: Path) -> None:
    (tmp_path / "visible.py").write_text("print('yes')\n", encoding="utf-8")
    indexer = ContextIndexer(
        workspace_root=tmp_path,
        roots=["."],
        db_path=".context_engine/index.db",
        max_file_bytes=1024,
        ignore_patterns=[],
    )
    indexer.scan_all()

    with indexer._connect() as conn:  # noqa: SLF001 - test-only direct DB seed
        conn.execute(
            "INSERT OR REPLACE INTO files(path, mtime, size, file_type, sha256, indexed_at) VALUES(?, ?, ?, ?, ?, ?)",
            (".hidden/secret.py", 0.0, 10, "code", "x" * 64, "2026-02-08T00:00:00Z"),
        )

    indexer.scan_all()
    paths = list(indexer.existing_paths())
    assert "visible.py" in paths
    assert ".hidden/secret.py" not in paths
