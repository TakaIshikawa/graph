"""Summarize namespace prefixes in unit tags and hashtag metadata values."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, flatten_values, get, metadata, sort_key, unit_id

_HASHTAG_RE = re.compile(r"(?<!\w)#([A-Za-z0-9][\w:/-]*)")


def summarize_unit_tag_prefixes(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    total_units = 0
    prefix_counts: Counter[str] = Counter()
    prefix_units: dict[str, set[str]] = defaultdict(set)
    display: dict[str, str] = {}
    examples: dict[str, list[dict[str, str]]] = defaultdict(list)
    tag_count = 0
    unprefixed_count = 0
    for unit in units:
        total_units += 1
        for tag in _tags(unit):
            tag_count += 1
            prefix = _prefix(tag)
            if not prefix:
                unprefixed_count += 1
                continue
            key = prefix.casefold()
            display.setdefault(key, prefix)
            prefix_counts[key] += 1
            prefix_units[key].add(unit_id(unit))
            if len(examples[key]) < sample_limit:
                examples[key].append({"unit_id": unit_id(unit), "tag": tag})
    prefixes = [
        {"prefix": display[key], "tag_count": count, "unit_count": len(prefix_units[key]), "examples": examples[key]}
        for key, count in prefix_counts.items()
    ]
    prefixes.sort(key=lambda row: (-row["tag_count"], sort_key(row["prefix"])))
    return {"total_units": total_units, "tag_count": tag_count, "unprefixed_count": unprefixed_count, "prefixes": prefixes}


def _tags(unit: Any) -> list[str]:
    values: list[Any] = []
    raw_tags = get(unit, "tags") or metadata(unit).get("tags")
    values.extend(flatten_values(raw_tags))
    for value in flatten_values(metadata(unit)):
        if isinstance(value, str):
            values.extend(match.group(1) for match in _HASHTAG_RE.finditer(value))
    return [field_value(value).lstrip("#") for value in values if field_value(value).lstrip("#")]


def _prefix(tag: str) -> str:
    positions = [pos for sep in ("/", ":", "-") if (pos := tag.find(sep)) > 0]
    return tag[: min(positions)] if positions else ""
