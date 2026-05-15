"""CSV export for unit metadata value frequencies."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "source_project",
    "source_entity_type",
    "metadata_key",
    "metadata_value",
    "unit_count",
    "unit_ids",
    "sample_titles",
]
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_metadata_value_frequency_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write grouped metadata value frequencies for units."""
    unit_list = list(units)
    rows = _frequency_rows(unit_list)
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


def _frequency_rows(units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str, str, str], dict[str, Any]] = defaultdict(lambda: {"unit_ids": set(), "titles": set()})

    for unit in units:
        metadata = unit.metadata if isinstance(unit.metadata, Mapping) else {}
        for raw_key, raw_value in metadata.items():
            key = _inline_text(raw_key)
            if not key:
                continue
            for value in _metadata_values(raw_value):
                value_text = _value_text(value)
                group_key = (
                    _field_value(unit.source_project) or "Unknown",
                    _field_value(unit.source_entity_type) or "Unknown",
                    key,
                    value_text,
                )
                groups[group_key]["unit_ids"].add(_field_value(unit.id))
                title = _inline_text(unit.title)
                if title:
                    groups[group_key]["titles"].add(title)

    rows: list[dict[str, str | int]] = []
    for (source_project, source_entity_type, metadata_key, metadata_value), payload in groups.items():
        unit_ids = sorted(payload["unit_ids"], key=_sort_key)
        titles = sorted(payload["titles"], key=_sort_key)
        rows.append(
            {
                "source_project": source_project,
                "source_entity_type": source_entity_type,
                "metadata_key": metadata_key,
                "metadata_value": metadata_value,
                "unit_count": len(unit_ids),
                "unit_ids": ";".join(unit_ids),
                "sample_titles": ";".join(titles[:3]),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            _sort_key(row["source_project"]),
            _sort_key(row["source_entity_type"]),
            _sort_key(row["metadata_key"]),
            _sort_key(row["metadata_value"]),
        ),
    )


def _metadata_values(value: Any) -> Iterable[Any]:
    if isinstance(value, list | tuple | set):
        for item in sorted(value, key=lambda item: _value_text(item)):
            yield item
        return
    yield value


def _value_text(value: Any) -> str:
    normalized = _json_safe(value)
    if isinstance(normalized, str):
        return _inline_text(normalized)
    if normalized is None:
        return ""
    if isinstance(normalized, bool):
        return "true" if normalized else "false"
    if isinstance(normalized, int | float):
        return str(normalized)
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=lambda item: _value_text(item))]
    return value


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
