"""CSV export for units with missing or blank content."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "unit_id",
    "title",
    "source_project",
    "source_entity_type",
    "content_state",
    "metadata_key_count",
    "tag_count",
    "created_at",
    "updated_at",
]
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_empty_content_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write units whose content is missing, empty, or whitespace."""
    unit_list = list(units)
    rows = _empty_content_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "empty_content_count": len(rows),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _empty_content_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for unit in units:
        state = _content_state(unit)
        if state is None:
            continue
        rows.append(
            {
                "unit_id": _unit_id(unit),
                "title": _field_value(_get(unit, "title")),
                "source_project": _field_value(_get(unit, "source_project")) or "Unknown",
                "source_entity_type": _field_value(_get(unit, "source_entity_type")) or "Unknown",
                "content_state": state,
                "metadata_key_count": len(_metadata(unit)),
                "tag_count": len(_unit_tags(unit)),
                "created_at": _field_value(_get(unit, "created_at")),
                "updated_at": _field_value(_get(unit, "updated_at")),
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            _sort_key(row["source_project"]),
            _sort_key(row["title"]),
            _sort_key(row["unit_id"]),
        ),
    )


def _content_state(unit: KnowledgeUnit | Mapping[str, Any]) -> str | None:
    content = _get(unit, "content", None)
    if content is None:
        return "missing"
    text = str(getattr(content, "value", content))
    if text == "":
        return "empty"
    if text.strip() == "":
        return "whitespace"
    return None


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _unit_tags(unit: KnowledgeUnit | Mapping[str, Any]) -> list[object]:
    tags = _get(unit, "tags")
    return list(tags) if isinstance(tags, Iterable) and not isinstance(tags, (str, bytes)) else []


def _unit_id(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "id")) or _field_value(_get(unit, "source_id"))


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


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
