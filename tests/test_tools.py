from __future__ import annotations

from pathlib import Path

from indexer import ContextIndexer
from search import CodeSearcher
from tools import code_search, context_recon, file_slice, project_overview, relevant_code


class DeterministicReranker:
    def rerank(self, query: str, candidates: list[dict], max_results: int) -> list[dict]:
        ranked = sorted(candidates, key=lambda c: c["base_score"], reverse=True)
        return [
            {
                "id": item["id"],
                "score": float(item["base_score"]),
                "rationale": "test",
            }
            for item in ranked[:max_results]
        ]


def _build_index(tmp_path: Path) -> tuple[ContextIndexer, CodeSearcher]:
    src = tmp_path / "src"
    src.mkdir(parents=True, exist_ok=True)

    (src / "analyzer.py").write_text(
        """
class TargetAnalyzer:
    def detect_bullet_holes(self, frame):
        mask = frame.apply_threshold()
        return mask.find_holes(min_size=3)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (src / "other.py").write_text("def noop():\n    return None\n", encoding="utf-8")
    hidden = tmp_path / ".hidden"
    hidden.mkdir(parents=True, exist_ok=True)
    (hidden / "secret.py").write_text("class Secret:\n    pass\n", encoding="utf-8")

    indexer = ContextIndexer(
        workspace_root=tmp_path,
        roots=["."],
        db_path=".context_engine/index.db",
        max_file_bytes=5 * 1024 * 1024,
        ignore_patterns=[],
    )
    indexer.scan_all()
    searcher = CodeSearcher(indexer)
    return indexer, searcher


def test_code_search_returns_hits(tmp_path: Path) -> None:
    indexer, searcher = _build_index(tmp_path)

    result = code_search.run(
        searcher=searcher,
        indexer=indexer,
        reranker=DeterministicReranker(),
        pattern="TargetAnalyzer",
        regex=False,
        case_sensitive=False,
        scope_paths=["src"],
        max_hits=50,
    )

    assert result["hits"]
    assert result["hits"][0]["path"] == "src/analyzer.py"


def test_file_slice_returns_hash_and_excerpt(tmp_path: Path) -> None:
    indexer, _ = _build_index(tmp_path)

    result = file_slice.run(indexer, path="src/analyzer.py", start_line=1, end_line=3)

    assert result["path"] == "src/analyzer.py"
    assert "TargetAnalyzer" in result["excerpt"]
    assert len(result["sha256"]) == 64


def test_file_slice_rejects_ignored_paths(tmp_path: Path) -> None:
    indexer, _ = _build_index(tmp_path)

    try:
        file_slice.run(indexer, path=".hidden/secret.py", start_line=1, end_line=2)
        assert False, "expected hidden path to be rejected"
    except PermissionError:
        pass


def test_project_overview_hides_dot_paths(tmp_path: Path) -> None:
    indexer, _ = _build_index(tmp_path)

    result = project_overview.run(indexer=indexer, scope_paths=None, max_depth=3)
    tree = result["tree"]
    assert ".hidden/" not in tree
    assert ".context_engine/" not in tree


def test_project_overview_ignores_hidden_scope_paths(tmp_path: Path) -> None:
    indexer, _ = _build_index(tmp_path)

    result = project_overview.run(indexer=indexer, scope_paths=[".hidden"], max_depth=3)
    assert result["tree"] == ""


def test_project_overview_marks_truncation_for_large_tree(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir(parents=True, exist_ok=True)
    for idx in range(700):
        (root / f"file_{idx}.txt").write_text("x\n", encoding="utf-8")

    indexer = ContextIndexer(
        workspace_root=tmp_path,
        roots=["."],
        db_path=".context_engine/index.db",
        max_file_bytes=1024,
        ignore_patterns=[],
    )
    indexer.scan_all()
    result = project_overview.run(indexer=indexer, scope_paths=None, max_depth=2)
    assert "tree_truncated" in result
    assert isinstance(result["tree_truncated"], bool)


def test_relevant_code_returns_ranked_snippets(tmp_path: Path) -> None:
    indexer, searcher = _build_index(tmp_path)

    result = relevant_code.run(
        indexer=indexer,
        searcher=searcher,
        reranker=DeterministicReranker(),
        query="How are bullet holes detected?",
        scope_paths=["src"],
        max_results=4,
        max_excerpt_lines=80,
        include_tests=False,
    )

    assert result["snippets"]
    top = result["snippets"][0]
    assert top["path"] == "src/analyzer.py"
    assert "detect_bullet_holes" in top["excerpt"]
    assert result["coverage"]["files_considered"] >= 1


def test_context_recon_returns_ranked_snippets(tmp_path: Path) -> None:
    indexer, searcher = _build_index(tmp_path)

    result = context_recon.run(
        indexer=indexer,
        searcher=searcher,
        reranker=DeterministicReranker(),
        query=None,
        question="How are bullet holes detected?",
        scope_paths=["src"],
        max_results=4,
        max_excerpt_lines=80,
        include_tests=False,
    )

    assert result["snippets"]
    assert result["retrieval_mode"] == "context_recon"
    top = result["snippets"][0]
    assert top["path"] == "src/analyzer.py"
    assert "detect_bullet_holes" in top["excerpt"]


def test_code_search_compacts_large_hit_lines(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir(parents=True, exist_ok=True)
    very_long = "x" * 1000
    (src / "long.py").write_text(f"value = '{very_long}'\n", encoding="utf-8")

    indexer = ContextIndexer(
        workspace_root=tmp_path,
        roots=["."],
        db_path=".context_engine/index.db",
        max_file_bytes=5 * 1024 * 1024,
        ignore_patterns=[],
    )
    indexer.scan_all()
    searcher = CodeSearcher(indexer)

    result = code_search.run(
        searcher=searcher,
        indexer=indexer,
        reranker=DeterministicReranker(),
        pattern="value",
        regex=False,
        case_sensitive=False,
        scope_paths=["src"],
        max_hits=10,
    )
    assert result["hits"]
    assert len(result["hits"][0]["text"]) <= 240
    assert "response_compacted" in result


def test_file_slice_compacts_large_excerpt(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir(parents=True, exist_ok=True)
    huge_line = "a" * 12000
    (src / "huge.py").write_text(f"{huge_line}\n", encoding="utf-8")

    indexer = ContextIndexer(
        workspace_root=tmp_path,
        roots=["."],
        db_path=".context_engine/index.db",
        max_file_bytes=5 * 1024 * 1024,
        ignore_patterns=[],
    )
    indexer.scan_all()

    result = file_slice.run(indexer=indexer, path="src/huge.py", start_line=1, end_line=1)
    assert result["response_compacted"] is True
    assert len(result["excerpt"]) <= 8_100
    assert result["effective_excerpt_lines"] == 200
