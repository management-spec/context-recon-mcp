from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Excerpt:
    path: str
    start_line: int
    end_line: int
    excerpt: str


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def clamp_window(center_line: int, total_lines: int, max_lines: int) -> tuple[int, int]:
    if total_lines <= 0:
        return 1, 1

    max_lines = max(1, max_lines)
    half = max_lines // 2

    start = max(1, center_line - half)
    end = min(total_lines, start + max_lines - 1)

    # readjust at file tail to preserve width
    start = max(1, end - max_lines + 1)
    return start, end


def slice_lines(path: Path, start_line: int, end_line: int, max_lines: int | None = None) -> Excerpt:
    lines = read_lines(path)
    total = len(lines)

    if total == 0:
        return Excerpt(path=path.as_posix(), start_line=1, end_line=1, excerpt="")

    start = max(1, start_line)
    end = max(start, end_line)

    if max_lines is not None and (end - start + 1) > max_lines:
        end = start + max_lines - 1

    end = min(end, total)
    excerpt = "\n".join(lines[start - 1 : end])
    return Excerpt(path=path.as_posix(), start_line=start, end_line=end, excerpt=excerpt)


def merge_overlapping_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []

    sorted_ranges = sorted(ranges)
    merged: list[tuple[int, int]] = [sorted_ranges[0]]

    for start, end in sorted_ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged
