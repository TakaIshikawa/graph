"""CSV export for unit-level metadata namespace coverage."""

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
    "unit_id",
    "title",
    "source_project",
    "namespace",
    "key_count",
    "populated_value_count",
    "empty_value_count",
    "list_value_count",
    "scalar_value_count",
    "keys",
]
_SEPARATOR_RE = re.compile(r"[.:/_]")
_WHITESPACE_RE = re.compile(r"\s+")


def export_units_to_metadata_namespace_matrix_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write per-unit metadata namespace coverage rows."""
    unit_list = list(units)
    rows = _matrix_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {"path": str(output_path), "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": output_path.stat().st_size}


def _matrix_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for unit in units:
        groups: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"keys": set(), "populated": 0, "empty": 0, "lists": 0, "scalars": 0}
        )
        for raw_key, value in _metadata(unit).items():
            key = _field_value(raw_key)
            if not key:
                continue
            namespace = _namespace(key)
            groups[namespace]["keys"].add(key)
            if _is_populated(value):
                groups[namespace]["populated"] += 1
            else:
                groups[namespace]["empty"] += 1
            if isinstance(value, list | tuple | set):
                groups[namespace]["lists"] += 1
            else:
                groups[namespace]["scalars"] += 1

        for namespace, group in groups.items():
            keys = sorted(group["keys"], key=_sort_key)
            rows.append(
                {
                    "unit_id": _unit_id(unit),
                    "title": _field_value(_get(unit, "title")),
                    "source_project": _field_value(_get(unit, "source_project")) or "Unknown",
                    "namespace": namespace,
                    "key_count": len(keys),
                    "populated_value_count": group["populated"],
                    "empty_value_count": group["empty"],
                    "list_value_count": group["lists"],
                    "scalar_value_count": group["scalars"],
                    "keys": "; ".join(keys),
                }
            )
    return sorted(rows, key=lambda row: (_sort_key(row["unit_id"]), _sort_key(row["title"]), _sort_key(row["namespace"])))


def _namespace(key: str) -> str:
    match = _SEPARATOR_RE.search(key)
    if match is None:
        return "unscoped"
    return key[: match.start()] or "unscoped"


def _is_populated(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(_inline_text(value))
    if isinstance(value, Mapping):
        return any(_is_populated(item) for item in value.values())
    if isinstance(value, list | tuple | set):
        return any(_is_populated(item) for item in value)
    return True


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _unit_id(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "id")) or _field_value(_get(unit, "source_id"))


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
