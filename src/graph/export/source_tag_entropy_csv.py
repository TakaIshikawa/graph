"""CSV export for source tag entropy summaries."""

from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["source", "unit_count", "tagged_unit_count", "unique_tag_count", "tag_entropy", "top_tag", "top_tag_share"]
_METADATA_TAG_KEYS = ("tags", "tag")
_WHITESPACE_RE = re.compile(r"\s+")


def export_source_tag_entropy_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write tag concentration summaries by source."""
    unit_list = list(units)
    rows = _entropy_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "source_count": len(rows),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _entropy_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    groups: dict[str, list[KnowledgeUnit | Mapping[str, Any]]] = defaultdict(list)
    for unit in units:
        groups[_field_value(_get(unit, "source_project")) or "Unknown"].append(unit)

    rows: list[dict[str, str | int]] = []
    for source, group_units in groups.items():
        counts: Counter[str] = Counter()
        displays: dict[str, str] = {}
        tagged_unit_count = 0
        for unit in group_units:
            tags = _unit_tags(unit)
            if tags:
                tagged_unit_count += 1
            for tag in tags:
                normalized = tag.casefold()
                counts[normalized] += 1
                displays.setdefault(normalized, tag)
        total_tags = sum(counts.values())
        top_tag = ""
        top_share = "0.00"
        if counts:
            top_normalized, top_count = sorted(
                counts.items(),
                key=lambda item: (-item[1], _sort_key(displays[item[0]])),
            )[0]
            top_tag = displays[top_normalized]
            top_share = f"{top_count / total_tags:.2f}"
        rows.append(
            {
                "source": source,
                "unit_count": len(group_units),
                "tagged_unit_count": tagged_unit_count,
                "unique_tag_count": len(counts),
                "tag_entropy": f"{_entropy(counts):.2f}",
                "top_tag": top_tag,
                "top_tag_share": top_share,
            }
        )
    return sorted(rows, key=lambda row: _sort_key(row["source"]))


def _entropy(counts: Counter[str]) -> float:
    if len(counts) < 2:
        return 0.0
    total = sum(counts.values())
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _unit_tags(unit: KnowledgeUnit | Mapping[str, Any]) -> list[str]:
    values: list[object] = []
    unit_tags = _get(unit, "tags")
    values.extend(_tag_values(unit_tags))
    metadata = _metadata(unit)
    for key in _METADATA_TAG_KEYS:
        values.extend(_tag_values(_casefold_get(metadata, key)))
    tags = {_field_value(value) for value in values if _field_value(value)}
    return sorted(tags, key=_sort_key)


def _tag_values(value: object) -> list[object]:
    if value is None or isinstance(value, bytes):
        return []
    if isinstance(value, str):
        return [part for part in value.split(",")]
    if isinstance(value, Mapping):
        return []
    if isinstance(value, list | tuple | set):
        return [item for value_item in value for item in _tag_values(value_item)]
    return [value]


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _casefold_get(mapping: Mapping[str, Any], key: str) -> object:
    for candidate_key, value in mapping.items():
        if _field_value(candidate_key).casefold() == key.casefold():
            return value
    return None


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
