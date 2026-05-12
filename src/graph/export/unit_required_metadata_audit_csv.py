"""CSV export for required unit metadata audit rows."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "unit_id",
    "title",
    "source_project",
    "source_entity_type",
    "metadata_key",
    "value_state",
    "available_metadata_keys",
]
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_required_metadata_audit_csv(
    units: Iterable[KnowledgeUnit],
    required_keys: Iterable[object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write rows for missing or empty required unit metadata."""
    unit_list = list(units)
    normalized_required_keys = _required_keys(required_keys)
    rows = _audit_rows(unit_list, normalized_required_keys)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "required_key_count": len(normalized_required_keys),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _audit_rows(units: list[KnowledgeUnit], required_keys: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for unit in units:
        metadata = unit.metadata if isinstance(unit.metadata, dict) else {}
        normalized_metadata = _normalized_metadata(metadata)
        available_metadata_keys = "; ".join(sorted(normalized_metadata, key=_sort_key))

        for metadata_key in required_keys:
            if metadata_key not in normalized_metadata:
                value_state = "missing"
            elif not _is_populated(normalized_metadata[metadata_key]):
                value_state = "empty"
            else:
                continue

            rows.append(
                {
                    "unit_id": _field_value(unit.id),
                    "title": _field_value(unit.title),
                    "source_project": _field_value(unit.source_project) or "Unknown",
                    "source_entity_type": _field_value(unit.source_entity_type) or "Unknown",
                    "metadata_key": metadata_key,
                    "value_state": value_state,
                    "available_metadata_keys": available_metadata_keys,
                }
            )

    return sorted(
        rows,
        key=lambda row: (
            _sort_key(row["source_project"]),
            _sort_key(row["source_entity_type"]),
            _sort_key(row["unit_id"]),
            _sort_key(row["metadata_key"]),
        ),
    )


def _render_csv(rows: list[dict[str, str]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _required_keys(required_keys: Iterable[object]) -> list[str]:
    keys = sorted({_inline_text(key) for key in required_keys if _inline_text(key)}, key=_sort_key)
    if not keys:
        raise ValueError("required_keys must include at least one non-empty key")
    return keys


def _normalized_metadata(metadata: dict) -> dict[str, object]:
    values: dict[str, object] = {}
    for key, value in metadata.items():
        normalized_key = _inline_text(key)
        if normalized_key and normalized_key not in values:
            values[normalized_key] = value
    return values


def _is_populated(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(_inline_text(value))
    if isinstance(value, list | tuple | set | dict):
        return len(value) > 0
    return True


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
