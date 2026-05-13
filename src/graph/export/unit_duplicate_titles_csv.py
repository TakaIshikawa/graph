"""CSV export for KnowledgeUnit title duplicates."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "normalized_title",
    "display_title",
    "duplicate_count",
    "unit_ids",
    "source_projects",
    "source_entity_types",
    "content_types",
]
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_duplicate_titles_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write duplicate title groups across units."""
    unit_list = list(units)
    rows = _duplicate_title_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "duplicate_group_count": len(rows),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _duplicate_title_rows(units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    groups: dict[str, list[KnowledgeUnit]] = defaultdict(list)
    for unit in units:
        normalized = _normalize_title(getattr(unit, "title", None))
        if normalized:
            groups[normalized].append(unit)

    rows: list[dict[str, str | int]] = []
    for normalized_title, group_units in groups.items():
        if len(group_units) < 2:
            continue
        rows.append(
            {
                "normalized_title": normalized_title,
                "display_title": _display_title(group_units),
                "duplicate_count": len(group_units),
                "unit_ids": _joined_unique(getattr(unit, "id", None) for unit in group_units),
                "source_projects": _joined_unique(getattr(unit, "source_project", None) for unit in group_units),
                "source_entity_types": _joined_unique(getattr(unit, "source_entity_type", None) for unit in group_units),
                "content_types": _joined_unique(getattr(unit, "content_type", None) for unit in group_units),
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            -int(row["duplicate_count"]),
            _sort_key(row["normalized_title"]),
        ),
    )


def _display_title(units: list[KnowledgeUnit]) -> str:
    titles = [_inline_text(getattr(unit, "title", None)) for unit in units]
    titles = [title for title in titles if title]
    if not titles:
        return ""
    return sorted(titles, key=lambda title: (-len(title), not any(char.islower() for char in title), _sort_key(title)))[0]


def _joined_unique(values: Iterable[object]) -> str:
    unique = {_field_value(value) for value in values}
    unique.discard("")
    return "; ".join(sorted(unique, key=_sort_key))


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _normalize_title(value: object) -> str:
    return _inline_text(value).casefold()


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
