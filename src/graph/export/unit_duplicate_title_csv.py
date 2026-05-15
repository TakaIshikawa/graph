"""CSV export for likely duplicate unit titles."""

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
    "normalized_title",
    "unit_count",
    "unit_ids",
    "source_projects",
    "source_ids",
    "content_types",
]
_SEPARATOR_RE = re.compile(r"[\s\-_:/|.,;]+")
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_duplicate_title_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write likely duplicate units grouped by normalized title."""
    unit_list = list(units)
    rows = _duplicate_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "duplicate_title_count": len(rows),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _duplicate_rows(units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    groups: dict[str, list[KnowledgeUnit]] = defaultdict(list)
    for unit in units:
        normalized_title = _normalized_title(unit)
        if normalized_title:
            groups[normalized_title].append(unit)

    rows: list[dict[str, str | int]] = []
    for normalized_title, title_units in groups.items():
        if len(title_units) < 2:
            continue
        rows.append(
            {
                "normalized_title": normalized_title,
                "unit_count": len(title_units),
                "unit_ids": _joined_unique(_field_value(unit.id) for unit in title_units),
                "source_projects": _joined_unique(_field_value(unit.source_project) for unit in title_units),
                "source_ids": _joined_unique(_field_value(unit.source_id) for unit in title_units),
                "content_types": _joined_unique(_field_value(unit.content_type) for unit in title_units),
            }
        )

    return sorted(rows, key=lambda row: (_sort_key(row["normalized_title"]), _sort_key(row["unit_ids"])))


def _normalized_title(unit: KnowledgeUnit) -> str:
    metadata = unit.metadata if isinstance(unit.metadata, Mapping) else {}
    title = _inline_text(metadata.get("canonical_title")) or _inline_text(unit.title)
    return _SEPARATOR_RE.sub(" ", title.casefold()).strip()


def _joined_unique(values: Iterable[str]) -> str:
    return "; ".join(sorted({value for value in values if value}, key=_sort_key))


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
