from __future__ import annotations

from pathlib import Path

from utils.ignore import IgnoreMatcher


def test_common_cache_folders_are_ignored(tmp_path: Path) -> None:
    matcher = IgnoreMatcher(root=tmp_path, extra_ignores=[])
    cache_file = tmp_path / "__pycache__" / "module.cpython-311.pyc"
    assert matcher.is_ignored(cache_file) is True


def test_system_top_level_dirs_only_ignored_for_broad_roots(tmp_path: Path) -> None:
    broad = IgnoreMatcher(root=Path("/"), extra_ignores=[])
    assert broad.is_ignored(Path("/System/Library/kext")) is True
    assert broad.is_ignored(Path("/Library/Caches/x")) is True

    project = IgnoreMatcher(root=tmp_path, extra_ignores=[])
    project_system_like = tmp_path / "System" / "MyDocs"
    assert project.is_ignored(project_system_like) is False
