"""CSV export for compact unit search snippets."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["unit_id", "title", "tags", "snippet", "snippet_length"]
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_search_snippet_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
    *,
    max_length: int = 160,
) -> str | dict[str, Any]:
    """Return or write compact searchable snippets for units."""
    if max_length < 1:
        raise ValueError("max_length must be positive")
    unit_list = list(units)
    rows = _snippet_rows(unit_list, max_length)
    text = _render_csv(rows)
    if path is None:
        return text
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {"path": str(output_path), "unit_count": len(unit_list), "rows_exported": len(rows), "max_length": max_length, "bytes_written": output_path.stat().st_size}


def _snippet_rows(units: list[KnowledgeUnit | Mapping[str, Any]], max_length: int) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for unit in units:
        tags = _tags(unit)
        raw = " ".join(part for part in (_field_value(_get(unit, "title")), " ".join(tags), _field_value(_get(unit, "content"))) if part)
        snippet = _truncate(_inline_text(raw), max_length)
        rows.append(
            {
                "unit_id": _unit_id(unit),
                "title": _field_value(_get(unit, "title")),
                "tags": "; ".join(tags),
                "snippet": snippet,
                "snippet_length": len(snippet),
            }
        )
    return sorted(rows, key=lambda row: _sort_key(row["unit_id"]))


def _truncate(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    if max_length <= 3:
        return text[:max_length]
    return text[: max_length - 3].rstrip() + "..."


def _tags(unit: KnowledgeUnit | Mapping[str, Any]) -> list[str]:
    value = _get(unit, "tags")
    return sorted({_field_value(item) for item in value if _field_value(item)}, key=_sort_key) if isinstance(value, list | tuple | set) else []


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
