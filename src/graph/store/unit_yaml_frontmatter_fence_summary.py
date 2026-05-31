"""Summarize leading YAML/TOML frontmatter fences in unit content."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, sort_key, unit_id


def summarize_unit_yaml_frontmatter_fences(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = units_with = closed = unclosed = yaml_count = toml_count = 0
    samples: list[dict[str, str | int | bool]] = []
    for unit in units:
        total += 1
        fence = _fence(str(get(unit, "content") or ""))
        if fence is None:
            continue
        marker, start_line, end_line, is_closed = fence
        units_with += 1
        closed += 1 if is_closed else 0
        unclosed += 1 if not is_closed else 0
        yaml_count += 1 if marker == "---" else 0
        toml_count += 1 if marker == "+++" else 0
        if len(samples) < limit:
            samples.append({"unit_id": unit_id(unit), "fence_marker": marker, "start_line": start_line, "end_line": end_line, "is_closed": is_closed})
    samples.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["start_line"])))
    return {"total_units": total, "units_with_frontmatter_fence": units_with, "closed_fence_count": closed, "unclosed_fence_count": unclosed, "yaml_fence_count": yaml_count, "toml_fence_count": toml_count, "samples": samples[:limit]}


def _fence(content: str) -> tuple[str, int, int, bool] | None:
    lines = content.splitlines()
    if not lines or lines[0].strip() not in {"---", "+++"}:
        return None
    marker = lines[0].strip()
    for index, line in enumerate(lines[1:], start=2):
        if line.strip() == marker:
            return marker, 1, index, True
    return marker, 1, 0, False
