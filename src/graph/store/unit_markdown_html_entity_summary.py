"""Summarize HTML entities in Markdown unit content."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import get, metadata, sort_key, unit_id

_ENTITY_RE = re.compile(r"&(?:#[0-9]+|#x[0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]+);")


def summarize_unit_markdown_html_entities(units: Iterable[Mapping[str, Any] | object]) -> dict[str, Any]:
    """Summarize named and numeric HTML entities by unit."""
    rows: list[dict[str, Any]] = []
    total_units = total_entities = 0
    entity_counts: Counter[str] = Counter()
    for index, unit in enumerate(units):
        total_units += 1
        uid = unit_id(unit) or str(index)
        counts = Counter(match.group(0) for match in _ENTITY_RE.finditer(_content(unit)))
        total = sum(counts.values())
        total_entities += total
        entity_counts.update(counts)
        top_entity = ""
        if counts:
            top_entity = sorted(counts.items(), key=lambda item: (-item[1], sort_key(item[0])))[0][0]
        rows.append({"unit_id": uid, "entity_count": total, "entity_counts": dict(sorted(counts.items(), key=lambda item: sort_key(item[0]))), "top_entity": top_entity})
    rows.sort(key=lambda row: sort_key(row["unit_id"]))
    return {
        "total_units": total_units,
        "total_entity_count": total_entities,
        "entity_counts": dict(sorted(entity_counts.items(), key=lambda item: sort_key(item[0]))),
        "units": rows,
    }


def _content(unit: Mapping[str, Any] | object) -> str:
    return str(get(unit, "content") or metadata(unit).get("content") or "")
