"""CSV export for source project and unit tag coverage."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "source_project",
    "tag",
    "unit_count",
    "content_type_count",
    "source_id_count",
    "example_unit_ids",
]
_TAG_KEYS = ("tags", "tag", "labels", "keywords")
_EXAMPLE_LIMIT = 5
_UNKNOWN = "Unknown"
_WHITESPACE_RE = re.compile(r"\s+")
_TAG_SPLIT_RE = re.compile(r"[;,]")


def export_unit_tag_matrix_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write source project/tag coverage rows for tagged units."""
    unit_list = list(units)
    rows = _tag_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "untagged_unit_count": len([unit for unit in unit_list if not _unit_tags(unit)]),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _tag_rows(units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str], list[KnowledgeUnit]] = defaultdict(list)
    for unit in units:
        source_project = _field_value(unit.source_project) or _UNKNOWN
        for tag in _unit_tags(unit):
            groups[(source_project, tag)].append(unit)

    rows: list[dict[str, str | int]] = []
    for (source_project, tag), tagged_units in groups.items():
        rows.append(
            {
                "source_project": source_project,
                "tag": tag,
                "unit_count": len(tagged_units),
                "content_type_count": len({_field_value(unit.content_type) for unit in tagged_units}),
                "source_id_count": len({_field_value(unit.source_id) for unit in tagged_units}),
                "example_unit_ids": "; ".join(
                    sorted({_field_value(unit.id) for unit in tagged_units}, key=_sort_key)[:_EXAMPLE_LIMIT]
                ),
            }
        )

    return sorted(rows, key=lambda row: (_sort_key(row["source_project"]), _sort_key(row["tag"])))


def _unit_tags(unit: KnowledgeUnit) -> list[str]:
    metadata = unit.metadata if isinstance(unit.metadata, Mapping) else {}
    tags: dict[str, str] = {}
    for key in _TAG_KEYS:
        for tag in _tag_values(metadata.get(key)):
            tags.setdefault(tag.casefold(), tag)
    return sorted(tags.values(), key=_sort_key)


def _tag_values(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values: Iterable[object] = _TAG_SPLIT_RE.split(value)
    elif isinstance(value, list | tuple | set):
        values = value
    else:
        values = [value]
    return [text for item in values if (text := _inline_text(item))]


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
