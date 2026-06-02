"""CSV export for per-unit tag co-occurrence."""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, flatten_values, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["tag_a", "tag_b", "unit_count", "unit_ids"]
_TAG_KEYS = {"tag", "tags", "label", "labels", "keyword", "keywords"}


def export_units_to_tag_cooccurrence_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    buckets: dict[tuple[str, str], set[str]] = defaultdict(set)
    for unit in unit_list:
        tags = sorted(set(_tags(unit)), key=sort_key)
        for tag_a, tag_b in itertools.combinations(tags, 2):
            buckets[(tag_a, tag_b)].add(unit_id(unit))
    rows = [
        {"tag_a": tag_a, "tag_b": tag_b, "unit_count": len(unit_ids), "unit_ids": "; ".join(sorted(unit_ids, key=sort_key))}
        for (tag_a, tag_b), unit_ids in buckets.items()
    ]
    rows.sort(key=lambda row: (sort_key(row["tag_a"]), sort_key(row["tag_b"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _tags(unit: Mapping[str, Any] | object) -> list[str]:
    tags: list[str] = []
    for container in (unit if isinstance(unit, Mapping) else {}, metadata(unit)):
        for key, value in container.items():
            if field_value(key).casefold() in _TAG_KEYS:
                for item in flatten_values(value):
                    for part in field_value(item).replace(";", ",").split(","):
                        tag = field_value(part)
                        if tag:
                            tags.append(tag.casefold())
    return tags
