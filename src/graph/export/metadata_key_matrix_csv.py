"""CSV export for unit metadata key presence and values."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable, Sequence
from datetime import date, datetime
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_BASE_FIELDNAMES = ["unit_id", "title", "source", "type"]
_WHITESPACE_RE = re.compile(r"\s+")


def export_metadata_key_matrix_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    keys: Sequence[str] | None = None,
    include_values: bool = False,
) -> str | dict[str, Any]:
    """Return or write a unit-by-metadata-key CSV matrix."""
    unit_list = sorted(list(units), key=_unit_sort_key)
    metadata_keys = [_inline_text(key) for key in keys] if keys is not None else _discover_keys(unit_list)
    rows = [_row(unit, metadata_keys, include_values=include_values) for unit in unit_list]
    text = _render_csv(rows, [*_BASE_FIELDNAMES, *metadata_keys])

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "metadata_key_count": len(metadata_keys),
        "include_values": include_values,
        "bytes_written": output_path.stat().st_size,
    }


def _discover_keys(units: list[KnowledgeUnit]) -> list[str]:
    keys: set[str] = set()
    for unit in units:
        keys.update(_inline_text(key) for key in (unit.metadata or {}) if _inline_text(key))
    return sorted(keys, key=_sort_key)


def _row(unit: KnowledgeUnit, keys: list[str], *, include_values: bool) -> dict[str, str]:
    metadata = unit.metadata or {}
    row = {
        "unit_id": _unit_id(unit),
        "title": _unit_title(unit),
        "source": _unit_source(unit),
        "type": _unit_type(unit),
    }
    for key in keys:
        if key not in metadata:
            row[key] = ""
        elif include_values:
            row[key] = _metadata_value_text(metadata[key])
        else:
            row[key] = "1"
    return row


def _metadata_value_text(value: Any) -> str:
    normalized = _json_value(value)
    if normalized is None:
        return ""
    if isinstance(normalized, str):
        return _inline_text(normalized)
    if isinstance(normalized, bool):
        return "true" if normalized else "false"
    if isinstance(normalized, int | float):
        return str(normalized)
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list | tuple | set):
        return [_json_value(item) for item in value]
    return str(value)


def _render_csv(rows: list[dict[str, str]], fieldnames: list[str]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _unit_id(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.id or unit.source_id)


def _unit_title(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.title) or _inline_text((unit.metadata or {}).get("title")) or _unit_id(unit)


def _unit_source(unit: KnowledgeUnit) -> str:
    return _field_value(unit.source_project) or "Unknown"


def _unit_type(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.source_entity_type) or _field_value(unit.content_type) or "Unknown"


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str]:
    return (_unit_title(unit).casefold(), _unit_source(unit).casefold(), _unit_id(unit))


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
