"""Summarize Markdown callout titles in unit content."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_CALLOUT_RE = re.compile(r"^\s*>\s*\[!(?P<type>[A-Za-z][A-Za-z0-9_-]*)\][+-]?\s*(?P<title>.*)$")


def summarize_unit_markdown_callout_titles(units: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    """Summarize blockquote callouts and optional titles."""
    unit_list = list(units)
    titled = untitled = 0
    title_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    samples: list[dict[str, str | int]] = []
    for index, unit in enumerate(unit_list):
        uid = unit_id(unit) or str(index)
        for line_number, callout_type, title in _callouts(_content(unit)):
            type_counts[callout_type] += 1
            if title:
                titled += 1
                title_counts[title] += 1
            else:
                untitled += 1
            samples.append({"unit_id": uid, "line_number": line_number, "callout_type": callout_type, "title": title})
    samples.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["callout_type"]), sort_key(row["title"])))
    return {
        "total_units": len(unit_list),
        "callouts_with_titles": titled,
        "callouts_without_titles": untitled,
        "title_counts": dict(sorted(title_counts.items(), key=lambda item: sort_key(item[0]))),
        "callout_type_counts": dict(sorted(type_counts.items(), key=lambda item: sort_key(item[0]))),
        "samples": samples[:sample_limit],
    }


def _content(unit: Mapping[str, Any] | object) -> str:
    return str(get(unit, "content") or metadata(unit).get("content") or "")


def _callouts(content: str) -> list[tuple[int, str, str]]:
    rows: list[tuple[int, str, str]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _CALLOUT_RE.match(line)
        if match:
            rows.append((line_number, match.group("type").casefold(), field_value(match.group("title")).strip()))
    return rows
