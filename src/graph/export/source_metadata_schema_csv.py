"""CSV export for per-source metadata schema coverage."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
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
    "populated_unit_count",
    "coverage_percent",
    "observed_type_names",
    "example_values",
]
_WHITESPACE_RE = re.compile(r"\s+")
_EXAMPLE_LIMIT = 3


def export_source_metadata_schema_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write metadata key coverage by source project and entity type."""
    unit_list = list(units)
    rows = _schema_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "schema_key_count": len(rows),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _schema_rows(units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str], list[KnowledgeUnit]] = defaultdict(list)
    for unit in units:
        groups[(_field_value(unit.source_project) or "Unknown", _field_value(unit.source_entity_type) or "Unknown")].append(
            unit
        )

    rows: list[dict[str, str | int]] = []
    for (source_project, source_entity_type), group_units in sorted(
        groups.items(), key=lambda item: (_sort_key(item[0][0]), _sort_key(item[0][1]))
    ):
        metadata_keys = sorted(
            {
                _inline_text(key)
                for unit in group_units
                for key in (unit.metadata if isinstance(unit.metadata, dict) else {})
                if _inline_text(key)
            },
            key=_sort_key,
        )
        for metadata_key in metadata_keys:
            values = [
                unit.metadata.get(metadata_key)
                for unit in group_units
                if isinstance(unit.metadata, dict) and metadata_key in unit.metadata
            ]
            populated_values = [value for value in values if _is_populated(value)]
            type_names = sorted({_type_name(value) for value in values}, key=_sort_key)
            examples = sorted({_example_value(value) for value in populated_values if _example_value(value)}, key=_sort_key)
            rows.append(
                {
                    "source_project": source_project,
                    "source_entity_type": source_entity_type,
                    "metadata_key": metadata_key,
                    "unit_count": len(group_units),
                    "populated_unit_count": len(populated_values),
                    "coverage_percent": _decimal(len(populated_values) * 100 / len(group_units)),
                    "observed_type_names": "; ".join(type_names),
                    "example_values": "; ".join(examples[:_EXAMPLE_LIMIT]),
                }
            )
    return rows


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _is_populated(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(_inline_text(value))
    if isinstance(value, list | tuple | set | dict):
        return len(value) > 0
    return True


def _type_name(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, tuple):
        return "tuple"
    if isinstance(value, set):
        return "set"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def _example_value(value: object) -> str:
    if isinstance(value, str):
        return _inline_text(value)
    if isinstance(value, bool | int | float) or value is None:
        return _inline_text(value)
    if isinstance(value, list | tuple):
        return _inline_text(json.dumps(list(value), sort_keys=True, default=str))
    if isinstance(value, set):
        return _inline_text(json.dumps(sorted(value, key=_sort_key), default=str))
    if isinstance(value, dict):
        return _inline_text(json.dumps(value, sort_keys=True, default=str))
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
