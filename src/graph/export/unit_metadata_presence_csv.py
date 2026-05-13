"""CSV export for per-unit metadata key presence."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Sequence
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "unit_id",
    "source_project",
    "source_entity_type",
    "metadata_key_count",
    "present_keys",
    "missing_keys",
    "completeness_ratio",
]
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_metadata_presence_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    keys: Sequence[str] | None = None,
) -> str | dict[str, Any]:
    """Return or write metadata key presence rows for units."""
    unit_list = list(units)
    selected_keys = _selected_keys(unit_list, keys)
    rows = _presence_rows(unit_list, selected_keys)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "metadata_key_count": len(selected_keys),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _presence_rows(units: list[KnowledgeUnit], keys: list[str]) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for unit in sorted(units, key=lambda unit: (_sort_key(getattr(unit, "source_project", None)), _sort_key(getattr(unit, "source_entity_type", None)), _sort_key(getattr(unit, "id", None)))):
        metadata = getattr(unit, "metadata", None)
        metadata_keys = set(metadata) if isinstance(metadata, dict) else set()
        present = sorted((key for key in keys if key in metadata_keys), key=_sort_key)
        missing = sorted((key for key in keys if key not in metadata_keys), key=_sort_key)
        rows.append(
            {
                "unit_id": _field_value(getattr(unit, "id", None)),
                "source_project": _field_value(getattr(unit, "source_project", None)) or "Unknown",
                "source_entity_type": _field_value(getattr(unit, "source_entity_type", None)) or "Unknown",
                "metadata_key_count": len(metadata_keys),
                "present_keys": "; ".join(present),
                "missing_keys": "; ".join(missing),
                "completeness_ratio": _format_ratio(len(present), len(keys)),
            }
        )
    return rows


def _selected_keys(units: list[KnowledgeUnit], keys: Sequence[str] | None) -> list[str]:
    if keys is not None:
        return list(dict.fromkeys(_inline_text(key) for key in keys if _inline_text(key)))
    discovered: set[str] = set()
    for unit in units:
        metadata = getattr(unit, "metadata", None)
        if isinstance(metadata, dict):
            discovered.update(_inline_text(key) for key in metadata if _inline_text(key))
    return sorted(discovered, key=_sort_key)


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _format_ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.00"
    return f"{numerator / denominator:.2f}"


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
