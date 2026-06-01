"""Summarize Pandoc-style fenced div blocks in unit content."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import get, metadata, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_DIV_RE = re.compile(r"^\s*:::\s*(?P<attrs>.*)$")
_CLASS_RE = re.compile(r"\.([A-Za-z0-9_-]+)")
_ID_RE = re.compile(r"#([A-Za-z0-9_-]+)")
_ATTR_KEY_RE = re.compile(r"(?<![.#])\b([A-Za-z_:][A-Za-z0-9_:.-]*)\s*=")


def summarize_unit_markdown_fenced_divs(units: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    """Summarize fenced div openers like ``::: {.class #id key=value}``."""
    unit_list = list(units)
    total_blocks = id_count = unmatched = 0
    units_with: set[str] = set()
    class_counts: Counter[str] = Counter()
    attr_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    for index, unit in enumerate(unit_list):
        uid = unit_id(unit) or str(index)
        depth = 0
        for line_number, attrs in _markers(_content(unit)):
            if not attrs:
                if depth:
                    depth -= 1
                else:
                    unmatched += 1
                continue
            depth += 1
            total_blocks += 1
            units_with.add(uid)
            classes = _CLASS_RE.findall(attrs)
            identifier_match = _ID_RE.search(attrs)
            identifier = identifier_match.group(1) if identifier_match else ""
            class_counts.update(classes)
            attr_counts.update(_ATTR_KEY_RE.findall(attrs))
            if identifier:
                id_count += 1
            samples.append({"unit_id": uid, "line_number": line_number, "classes": sorted(classes, key=sort_key), "identifier": identifier})
    samples.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"])))
    return {
        "total_units": len(unit_list),
        "total_blocks": total_blocks,
        "units_with_fenced_divs": len(units_with),
        "class_counts": dict(sorted(class_counts.items(), key=lambda item: sort_key(item[0]))),
        "id_count": id_count,
        "attribute_key_counts": dict(sorted(attr_counts.items(), key=lambda item: sort_key(item[0]))),
        "unmatched_closing_markers": unmatched,
        "samples": samples[:sample_limit],
    }


def _content(unit: Mapping[str, Any] | object) -> str:
    return str(get(unit, "content") or metadata(unit).get("content") or "")


def _markers(content: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _DIV_RE.match(line)
        if match:
            rows.append((line_number, match.group("attrs").strip()))
    return rows
