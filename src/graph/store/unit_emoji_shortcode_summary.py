"""Summarize emoji shortcode usage by unit source."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key

_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_URL_RE = re.compile(r"https?://[^\s<>()\[\]\"']+", re.IGNORECASE)
_SHORTCODE_RE = re.compile(r":([A-Za-z0-9_+-]+):")
_VALID_SHORTCODE_RE = re.compile(r"^[a-z0-9_+-]+$")


def summarize_unit_emoji_shortcodes(units: Iterable[Any]) -> dict[str, Any]:
    """Return deterministic emoji shortcode counts grouped by source."""
    total_units = 0
    data: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "unit_count": 0,
            "units_with_shortcodes": 0,
            "shortcode_count": 0,
            "invalid_shortcode_count": 0,
            "counts": Counter(),
        }
    )

    for unit in units:
        total_units += 1
        source = _source(unit)
        row = data[source]
        row["unit_count"] += 1
        valid, invalid = _shortcodes(_content(unit))
        if valid or invalid:
            row["units_with_shortcodes"] += 1
        row["shortcode_count"] += len(valid)
        row["invalid_shortcode_count"] += invalid
        row["counts"].update(valid)

    rows = []
    for source in sorted(data, key=sort_key):
        counts = data[source]["counts"]
        rows.append(
            {
                "source": source,
                "unit_count": data[source]["unit_count"],
                "units_with_shortcodes": data[source]["units_with_shortcodes"],
                "shortcode_count": data[source]["shortcode_count"],
                "unique_shortcode_count": len(counts),
                "most_common_shortcode": _most_common(counts),
                "invalid_shortcode_count": data[source]["invalid_shortcode_count"],
            }
        )
    return {"total_units": total_units, "sources": rows}


def _content(unit: Any) -> str:
    value = get(unit, "content")
    if value is None:
        value = metadata(unit).get("content")
    return "" if value is None else str(value)


def _source(unit: Any) -> str:
    return field_value(get(unit, "source") or metadata(unit).get("source") or "unknown")


def _shortcodes(content: str) -> tuple[list[str], int]:
    valid: list[str] = []
    invalid = 0
    in_fence = False
    for line in content.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in _SHORTCODE_RE.finditer(_URL_RE.sub("", line)):
            name = match.group(1)
            normalized = name.casefold()
            if name == normalized and _VALID_SHORTCODE_RE.match(name):
                valid.append(normalized)
            else:
                invalid += 1
    return valid, invalid


def _most_common(counts: Counter[str]) -> str:
    if not counts:
        return ""
    return min(counts, key=lambda shortcode: (-counts[shortcode], sort_key(shortcode)))
