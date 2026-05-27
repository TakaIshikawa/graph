"""Summarize likely content encoding issues by source."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, unit_id

_MOJIBAKE_MARKERS = ("Ã", "Â", "â€™", "â€œ", "â€", "�")
_SOURCE_KEYS = ("source", "source_project")


def summarize_unit_content_encoding_issues(
    units: Iterable[Any], *, sample_limit: int = 5
) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = 0
    affected_unit_ids: set[str] = set()
    groups: dict[str, dict[str, Any]] = {}

    for index, unit in enumerate(units):
        total += 1
        uid = unit_id(unit) or str(index)
        source = _source(unit)
        content = "" if get(unit, "content") is None else str(get(unit, "content"))
        mojibake_count = sum(content.count(marker) for marker in _MOJIBAKE_MARKERS)
        replacement_count = content.count("\ufffd")
        control_count = sum(1 for char in content if ord(char) < 32 and char not in "\n\r\t")

        if not (mojibake_count or replacement_count or control_count):
            continue

        affected_unit_ids.add(uid)
        group = groups.setdefault(
            source,
            {
                "source": source,
                "affected_unit_count": 0,
                "mojibake_issue_count": 0,
                "replacement_character_count": 0,
                "control_character_count": 0,
                "representative_unit_ids": [],
            },
        )
        group["affected_unit_count"] += 1
        group["mojibake_issue_count"] += mojibake_count
        group["replacement_character_count"] += replacement_count
        group["control_character_count"] += control_count
        if len(group["representative_unit_ids"]) < limit:
            group["representative_unit_ids"].append(uid)

    rows = []
    for source in sorted(groups, key=sort_key):
        group = groups[source]
        rows.append(
            {
                "source": group["source"],
                "affected_unit_count": group["affected_unit_count"],
                "mojibake_issue_count": group["mojibake_issue_count"],
                "replacement_character_count": group["replacement_character_count"],
                "control_character_count": group["control_character_count"],
                "representative_unit_ids": group["representative_unit_ids"],
            }
        )

    return {
        "total_units": total,
        "affected_unit_count": len(affected_unit_ids),
        "source_summaries": rows,
    }


def _source(unit: Any) -> str:
    meta = metadata(unit)
    for key in _SOURCE_KEYS:
        value = field_value(get(unit, key))
        if value:
            return value
        value = field_value(meta.get(key))
        if value:
            return value
    return "unknown"
