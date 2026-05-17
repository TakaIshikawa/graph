"""CSV export for units with weak titles."""

from __future__ import annotations

import csv
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "unit_id",
    "title",
    "source_project",
    "issue_count",
    "issues",
    "title_length",
    "duplicate_count",
]
_GENERIC_TITLES = {"note", "notes", "title", "untitled", "document", "page", "new note", "unknown"}
_LONG_TITLE_LENGTH = 120
_URL_RE = re.compile(r"^(?:https?://|www\.)\S+$", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_title_quality_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write units with at least one title quality issue."""
    unit_list = list(units)
    rows = _quality_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "weak_title_count": len(rows),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _quality_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    title_counts = Counter(
        normalized for normalized in (_normalize_title(_get(unit, "title")) for unit in units) if normalized
    )

    rows: list[dict[str, str | int]] = []
    for unit in units:
        unit_id = _unit_id(unit)
        title = _field_value(_get(unit, "title"))
        normalized = _normalize_title(title)
        duplicate_count = title_counts.get(normalized, 0) if normalized else 0
        issues = _issues(unit_id, title, normalized, duplicate_count)
        if not issues:
            continue
        rows.append(
            {
                "unit_id": unit_id,
                "title": title,
                "source_project": _field_value(_get(unit, "source_project")) or "Unknown",
                "issue_count": len(issues),
                "issues": "; ".join(issues),
                "title_length": len(title),
                "duplicate_count": duplicate_count,
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


def _issues(unit_id: str, title: str, normalized: str, duplicate_count: int) -> list[str]:
    issues: list[str] = []
    if not normalized:
        issues.append("missing")
    elif normalized in _GENERIC_TITLES:
        issues.append("generic")
    if duplicate_count > 1:
        issues.append("duplicate")
    if len(title) > _LONG_TITLE_LENGTH:
        issues.append("long")
    if title and _URL_RE.match(title):
        issues.append("url_like")
    if normalized and normalized == _normalize_title(unit_id):
        issues.append("same_as_id")
    return issues


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


def _normalize_title(value: object) -> str:
    return _inline_text(value).casefold()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
