"""CSV export for attachment-like unit metadata."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "unit_id",
    "source_project",
    "source_entity_type",
    "metadata_key",
    "value_type",
    "value",
]
_ATTACHMENT_KEYS = {
    "url",
    "urls",
    "link",
    "links",
    "file",
    "files",
    "path",
    "paths",
    "attachment",
    "attachments",
}
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_attachment_inventory_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write one row per attachment-like metadata value."""
    unit_list = list(units)
    rows = _attachment_rows(unit_list)
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


def _attachment_rows(units: list[KnowledgeUnit]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for unit in units:
        metadata = unit.metadata if isinstance(unit.metadata, Mapping) else {}
        for key, value in metadata.items():
            key_text = _field_value(key)
            if _normalized_key(key_text) not in _ATTACHMENT_KEYS:
                continue
            for item in _iter_values(value):
                text = _inline_text(item)
                if not text:
                    continue
                rows.append(
                    {
                        "unit_id": _field_value(unit.id),
                        "source_project": _field_value(unit.source_project) or "Unknown",
                        "source_entity_type": _field_value(unit.source_entity_type) or "Unknown",
                        "metadata_key": key_text,
                        "value_type": _value_type(text),
                        "value": text,
                    }
                )
    return sorted(
        rows,
        key=lambda row: (
            _sort_key(row["unit_id"]),
            _sort_key(row["metadata_key"]),
            _sort_key(row["value"]),
        ),
    )


def _iter_values(value: object) -> list[object]:
    if isinstance(value, list | tuple | set):
        return list(value)
    return [value]


def _value_type(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme.casefold() in {"http", "https"} and parsed.netloc:
        return "url"
    if value.startswith(("/", "./", "../", "~")) or "/" in value or "\\" in value:
        return "path"
    return "text"


def _render_csv(rows: list[dict[str, str]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _normalized_key(value: str) -> str:
    return value.casefold().replace("-", "_").replace(" ", "_")


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
