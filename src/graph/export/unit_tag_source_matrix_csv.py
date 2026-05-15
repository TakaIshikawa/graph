"""CSV export for unit tag counts crossed with source metadata."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["tag", "source_type", "source_label", "unit_count", "source_observation_count"]
_UNKNOWN = "unknown"
_LABEL_KEYS = ("label", "name", "title", "source_label", "source_name", "source_title")
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_tag_source_matrix_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write tag/source rows with unit and source observation counts."""
    unit_list = list(units)
    rows = _matrix_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _matrix_rows(units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    unit_ids: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    observation_counts: dict[tuple[str, str, str], int] = defaultdict(int)

    for unit in sorted(units, key=_unit_sort_key):
        unit_id = _unit_id(unit)
        for tag in _unit_tags(unit):
            for source_type, source_label in _source_observations(unit):
                key = (tag, source_type, source_label)
                unit_ids[key].add(unit_id)
                observation_counts[key] += 1

    return [
        {
            "tag": tag,
            "source_type": source_type,
            "source_label": source_label,
            "unit_count": len(unit_ids[key]),
            "source_observation_count": observation_counts[key],
        }
        for key in sorted(
            observation_counts,
            key=lambda item: (_sort_key(item[0]), _sort_key(item[1]), _sort_key(item[2])),
        )
        for tag, source_type, source_label in [key]
    ]


def _unit_tags(unit: KnowledgeUnit) -> list[str]:
    tags = sorted({_inline_text(tag) for tag in unit.tags if _inline_text(tag)}, key=_sort_key)
    return tags or [_UNKNOWN]


def _source_observations(unit: KnowledgeUnit) -> list[tuple[str, str]]:
    metadata = unit.metadata if isinstance(unit.metadata, Mapping) else {}
    raw_sources: list[object] = []
    if "source" in metadata:
        raw_sources.extend(_flat_values(metadata.get("source")))
    if "sources" in metadata:
        raw_sources.extend(_flat_values(metadata.get("sources")))
    return [_source_observation(source) for source in raw_sources] or [(_UNKNOWN, _UNKNOWN)]


def _flat_values(value: object) -> list[object]:
    if isinstance(value, list | tuple | set):
        return [item for entry in value for item in _flat_values(entry)]
    return [value]


def _source_observation(value: object) -> tuple[str, str]:
    if isinstance(value, Mapping):
        source_type = _metadata_text(value, ("source_type", "type", "kind")) or _UNKNOWN
        source_label = _metadata_text(value, _LABEL_KEYS) or _UNKNOWN
        return source_type, source_label
    text = _inline_text(value)
    return (_UNKNOWN, text or _UNKNOWN)


def _metadata_text(metadata: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        text = _inline_text(metadata.get(key))
        if text:
            return text
    return ""


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _unit_id(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.id) or _inline_text(unit.source_id)


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[tuple[str, str], tuple[str, str]]:
    return (_sort_key(_unit_id(unit)), _sort_key(unit.title))
