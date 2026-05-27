"""Summarize inline Markdown code spans across units."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, unit_id

_INLINE_RE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")


def summarize_unit_inline_code_usage(units: Iterable[Any], *, sample_limit: int = 5, high_density_threshold: float = 0.08) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total_units = units_with_inline_code = span_count = 0
    snippets: Counter[str] = Counter()
    high_density: list[dict[str, Any]] = []
    for index, unit in enumerate(units):
        total_units += 1
        content = str(get(unit, "content") or "")
        spans = _spans(content)
        if spans:
            units_with_inline_code += 1
        span_count += len(spans)
        snippets.update(spans)
        words = max(1, len(re.findall(r"\S+", _without_fences(content))))
        density = len(spans) / words
        if spans and density >= high_density_threshold:
            high_density.append({"unit_id": unit_id(unit) or str(index), "title": _title(unit), "inline_code_count": len(spans), "density": round(density, 4)})
    common = [{"snippet": text, "count": count} for text, count in sorted(snippets.items(), key=lambda item: (-item[1], sort_key(item[0])))[:limit]]
    return {
        "total_units": total_units,
        "units_with_inline_code": units_with_inline_code,
        "inline_code_span_count": span_count,
        "distinct_code_tokens": len(snippets),
        "common_snippets": common,
        "high_density_units": sorted(high_density, key=lambda row: (-float(row["density"]), sort_key(row["unit_id"])))[:limit],
    }


def _spans(content: str) -> list[str]:
    return [field_value(match.group(1)) for match in _INLINE_RE.finditer(_without_fences(content))]


def _without_fences(content: str) -> str:
    lines: list[str] = []
    in_fence = False
    for line in content.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if not in_fence:
            lines.append(line)
    return "\n".join(lines)


def _title(unit: Any) -> str:
    return field_value(get(unit, "title") or metadata(unit).get("title"))
