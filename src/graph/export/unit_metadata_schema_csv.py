"""CSV export for unit metadata key usage by source and entity type."""

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
    "source_project",
    "source_entity_type",
    "metadata_key",
    "unit_count",
    "present_count",
    "coverage_percent",
    "value_types",
    "sample_values",
]
_WHITESPACE_RE = re.compile(r"\s+")
_SAMPLE_LIMIT = 3


def export_unit_metadata_schema_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    min_count: int = 1,
) -> str | dict[str, Any]:
    """Return or write metadata key usage grouped by source project and entity type."""
    _validate_min_count(min_count)

    unit_list = list(units)
    rows = _schema_rows(unit_list, min_count=min_count)
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
        "min_count": min_count,
        "bytes_written": output_path.stat().st_size,
    }


def _schema_rows(
    units: list[KnowledgeUnit],
    *,
    min_count: int,
) -> list[dict[str, str | int]]:
    group_counts: Counter[tuple[str, str]] = Counter()
    present_counts: Counter[tuple[str, str, str]] = Counter()
    value_types: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    sample_values: dict[tuple[str, str, str], set[str]] = defaultdict(set)

    for unit in sorted(units, key=_unit_sort_key):
        source_project = _unit_source(unit)
        entity_type = _unit_type(unit)
        group_counts[(source_project, entity_type)] += 1

        metadata = unit.metadata if isinstance(unit.metadata, dict) else {}
        seen_keys: set[str] = set()
        for raw_key, value in metadata.items():
            key = _metadata_key(raw_key)
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            row_key = (source_project, entity_type, key)
            present_counts[row_key] += 1
            value_types[row_key].add(_value_type_name(value))
            sample = _sample_value(value)
            if sample:
                sample_values[row_key].add(sample)

    rows: list[dict[str, str | int]] = []
    for source_project, entity_type, key in sorted(
        present_counts,
        key=lambda item: (_sort_key(item[0]), _sort_key(item[1]), _sort_key(item[2])),
    ):
        present_count = present_counts[(source_project, entity_type, key)]
        if present_count < min_count:
            continue
        unit_count = group_counts[(source_project, entity_type)]
        rows.append(
            {
                "source_project": source_project,
                "source_entity_type": entity_type,
                "metadata_key": key,
                "unit_count": unit_count,
                "present_count": present_count,
                "coverage_percent": _decimal((present_count / unit_count) * 100),
                "value_types": "; ".join(sorted(value_types[(source_project, entity_type, key)], key=_sort_key)),
                "sample_values": "; ".join(
                    sorted(sample_values[(source_project, entity_type, key)], key=_sort_key)[:_SAMPLE_LIMIT]
                ),
            }
        )
    return rows


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _validate_min_count(min_count: int) -> None:
    if not isinstance(min_count, int) or isinstance(min_count, bool) or min_count < 1:
        raise ValueError("min_count must be a positive integer")


def _unit_source(unit: KnowledgeUnit) -> str:
    return _field_value(unit.source_project) or "Unknown"


def _unit_type(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.source_entity_type) or "Unknown"


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[tuple[str, str], tuple[str, str], tuple[str, str]]:
    return (
        _sort_key(_unit_source(unit)),
        _sort_key(_unit_type(unit)),
        _sort_key(unit.id or unit.source_id),
    )


def _metadata_key(value: object) -> str:
    return _inline_text(value)


def _value_type_name(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def _sample_value(value: object) -> str:
    if isinstance(value, list):
        return _inline_text(", ".join(_inline_text(item) for item in value))
    if isinstance(value, dict):
        return _inline_text(value)
    return _inline_text(value)


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)


def _decimal(value: float) -> str:
    return f"{value:.2f}"
