"""CSV export for collection composition across source projects."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "collection",
    "source_project",
    "unit_count",
    "source_entity_types",
    "top_tags",
]
_DEFAULT_COLLECTION_KEYS = ("collection", "collections", "folder", "board", "project", "list")
_UNASSIGNED = "Unassigned"
_TOP_TAG_LIMIT = 5
_WHITESPACE_RE = re.compile(r"\s+")


def export_source_collection_mix_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    collection_keys: Iterable[str] | None = None,
) -> str | dict[str, Any]:
    """Return or write collection composition grouped by collection and source project."""
    keys = _collection_key_set(collection_keys)

    unit_list = list(units)
    rows = _mix_rows(unit_list, collection_keys=keys)
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
        "collection_key_count": len(keys),
        "bytes_written": output_path.stat().st_size,
    }


def _mix_rows(
    units: list[KnowledgeUnit],
    *,
    collection_keys: set[str],
) -> list[dict[str, str | int]]:
    unit_counts: Counter[tuple[str, str]] = Counter()
    entity_type_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    tag_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    for unit in sorted(units, key=_unit_sort_key):
        source_project = _unit_source(unit)
        entity_type = _unit_type(unit)
        collections = _unit_collections(unit, collection_keys=collection_keys)
        if not collections:
            collections = [_UNASSIGNED]

        for collection in collections:
            group_key = (collection, source_project)
            unit_counts[group_key] += 1
            entity_type_counts[group_key][entity_type] += 1
            tag_counts[group_key].update(_unit_tags(unit))

    rows: list[dict[str, str | int]] = []
    for collection, source_project in sorted(
        unit_counts,
        key=lambda item: (_sort_key(item[0]), _sort_key(item[1])),
    ):
        group_key = (collection, source_project)
        rows.append(
            {
                "collection": collection,
                "source_project": source_project,
                "unit_count": unit_counts[group_key],
                "source_entity_types": _counter_summary(entity_type_counts[group_key]),
                "top_tags": _top_tags(tag_counts[group_key]),
            }
        )
    return rows


def _unit_collections(unit: KnowledgeUnit, *, collection_keys: set[str]) -> list[str]:
    metadata = unit.metadata if isinstance(unit.metadata, dict) else {}
    collections: set[str] = set()
    for key, value in metadata.items():
        if _inline_text(key).casefold() not in collection_keys:
            continue
        collections.update(_collection_values(value))
    return sorted(collections, key=_sort_key)


def _collection_values(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, dict):
        return {_inline_text(value)} if _inline_text(value) else set()
    if isinstance(value, list | tuple | set):
        values: set[str] = set()
        for item in value:
            text = _inline_text(item)
            if text:
                values.add(text)
        return values
    text = _inline_text(value)
    return {text} if text else set()


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _collection_key_set(collection_keys: Iterable[str] | None) -> set[str]:
    keys = _DEFAULT_COLLECTION_KEYS if collection_keys is None else collection_keys
    return {_inline_text(key).casefold() for key in keys if _inline_text(key)}


def _unit_source(unit: KnowledgeUnit) -> str:
    return _field_value(unit.source_project) or "Unknown"


def _unit_type(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.source_entity_type) or "Unknown"


def _unit_tags(unit: KnowledgeUnit) -> list[str]:
    if not isinstance(unit.tags, list):
        return []
    return [tag for tag in (_inline_text(tag) for tag in unit.tags) if tag]


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[tuple[str, str], tuple[str, str]]:
    return (_sort_key(_unit_source(unit)), _sort_key(unit.id or unit.source_id))


def _counter_summary(counter: Counter[str]) -> str:
    return "; ".join(
        f"{key}:{count}"
        for key, count in sorted(counter.items(), key=lambda item: (-item[1], _sort_key(item[0])))
    )


def _top_tags(counter: Counter[str]) -> str:
    return "; ".join(
        f"{key}:{count}"
        for key, count in sorted(counter.items(), key=lambda item: (-item[1], _sort_key(item[0])))[:_TOP_TAG_LIMIT]
    )


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
