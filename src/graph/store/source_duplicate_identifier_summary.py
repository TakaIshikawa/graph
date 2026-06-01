"""Summarize duplicate source identifiers."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_DEFAULT_KEYS = ("source_id", "external_id", "identifier", "guid", "uri", "doi", "isbn")


def summarize_source_duplicate_identifiers(
    sources: Iterable[Mapping[str, Any] | object], identifier_keys: Sequence[str] | None = None, sample_limit: int = 5
) -> dict[str, Any]:
    source_list = list(sources)
    keys = tuple(identifier_keys or _DEFAULT_KEYS)
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    key_counts: Counter[str] = Counter()
    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        data = metadata(source)
        for key in keys:
            value = field_value(get(source, key)) or field_value(data.get(key))
            if value:
                groups[(key, value)].append(sid)
                key_counts[key] += 1
    duplicates = [
        {"identifier_key": key, "identifier_value": value, "source_count": len(ids), "source_ids": sorted(ids, key=sort_key)}
        for (key, value), ids in groups.items()
        if len(ids) > 1
    ]
    duplicates.sort(key=lambda row: (sort_key(row["identifier_key"]), sort_key(row["identifier_value"])))
    limit = max(0, sample_limit)
    return {
        "total_sources": len(source_list),
        "duplicate_identifier_count": len(duplicates),
        "duplicate_groups": duplicates,
        "key_counts": dict(sorted(key_counts.items())),
        "samples": duplicates[:limit],
    }
