"""CSV export for blank field coverage by source and entity type."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_CORE_FIELDS = ("title", "content", "source_id", "created_at", "updated_at")
_FIELDNAMES = ["source_project", "source_entity_type", "field_name", "unit_count", "blank_count", "present_count", "blank_percent"]
_WHITESPACE_RE = re.compile(r"\s+")


def export_source_field_nullability_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
    *,
    metadata_keys: Iterable[str] | None = None,
    min_blank_percent: float = 0.0,
) -> str | dict[str, Any]:
    """Return or write blank coverage for core fields and selected metadata keys."""
    min_blank_percent = _percent(min_blank_percent)
    unit_list = list(units)
    rows = _nullability_rows(unit_list, tuple(metadata_keys or ()), min_blank_percent)
    text = _render_csv(rows)
    if path is None:
        return text
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {"path": str(output_path), "unit_count": len(unit_list), "rows_exported": len(rows), "min_blank_percent": _decimal(min_blank_percent), "bytes_written": output_path.stat().st_size}


def _nullability_rows(units: list[KnowledgeUnit | Mapping[str, Any]], metadata_keys: tuple[str, ...], min_blank_percent: float) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str], list[KnowledgeUnit | Mapping[str, Any]]] = defaultdict(list)
    for unit in units:
        groups[(_field_value(_get(unit, "source_project")) or "Unknown", _field_value(_get(unit, "source_entity_type")) or "Unknown")].append(unit)
    rows: list[dict[str, str | int]] = []
    fields = [*_CORE_FIELDS, *(f"metadata:{key}" for key in metadata_keys)]
    for (source_project, source_entity_type), grouped_units in groups.items():
        for field in fields:
            blank_count = sum(1 for unit in grouped_units if _is_blank(_field(unit, field)))
            blank_percent = (blank_count / len(grouped_units) * 100.0) if grouped_units else 0.0
            if blank_percent < min_blank_percent:
                continue
            rows.append({"source_project": source_project, "source_entity_type": source_entity_type, "field_name": field, "unit_count": len(grouped_units), "blank_count": blank_count, "present_count": len(grouped_units) - blank_count, "blank_percent": _decimal(blank_percent)})
    return sorted(rows, key=lambda row: (_sort_key(row["source_project"]), _sort_key(row["source_entity_type"]), _sort_key(row["field_name"])))


def _field(unit: KnowledgeUnit | Mapping[str, Any], field: str) -> object:
    if field.startswith("metadata:"):
        return _metadata(unit).get(field.removeprefix("metadata:"))
    return _get(unit, field)


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return _inline_text(value) == ""
    if isinstance(value, Mapping):
        return len(value) == 0
    if isinstance(value, list | tuple | set):
        return len(value) == 0
    return False


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _percent(value: object) -> float:
    try:
        percent = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("min_blank_percent must be between 0 and 100") from exc
    if percent < 0.0 or percent > 100.0:
        raise ValueError("min_blank_percent must be between 0 and 100")
    return percent


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


def _decimal(value: float) -> str:
    return f"{value:.2f}"
