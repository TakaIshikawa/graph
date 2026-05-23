"""CSV export for likely unit metadata key collisions."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["unit_id", "title", "normalized_key", "original_keys", "value_count"]
_SEPARATOR_RE = re.compile(r"[\s_-]+")
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_metadata_key_collision_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write metadata keys that collide after normalization."""
    unit_list = list(units)
    rows = _collision_rows(unit_list)
    text = _render_csv(rows)
    if path is None:
        return text
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {"path": str(output_path), "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": output_path.stat().st_size}


def _collision_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for unit in units:
        groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"keys": set(), "values": []})
        for raw_key, value in _metadata(unit).items():
            key = _field_value(raw_key)
            if not key:
                continue
            normalized = _normalize_key(key)
            groups[normalized]["keys"].add(key)
            groups[normalized]["values"].extend(_flatten(value))
        for normalized, group in groups.items():
            if len(group["keys"]) < 2:
                continue
            rows.append(
                {
                    "unit_id": _unit_id(unit),
                    "title": _field_value(_get(unit, "title")),
                    "normalized_key": normalized,
                    "original_keys": "; ".join(sorted(group["keys"], key=_sort_key)),
                    "value_count": len([value for value in group["values"] if _field_value(value)]),
                }
            )
    return sorted(rows, key=lambda row: (_sort_key(row["unit_id"]), _sort_key(row["normalized_key"])))


def _normalize_key(key: str) -> str:
    return _SEPARATOR_RE.sub("", key).casefold()


def _flatten(value: object) -> list[object]:
    if isinstance(value, Mapping):
        return [item for child in value.values() for item in _flatten(child)]
    if isinstance(value, list | tuple | set):
        return [item for child in value for item in _flatten(child)]
    return [value]


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
