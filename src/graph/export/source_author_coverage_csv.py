"""CSV export for source author metadata coverage."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "source_project",
    "unit_count",
    "units_with_author",
    "author_coverage_percent",
    "distinct_authors",
    "top_authors",
]
_AUTHOR_KEYS = {"author", "authors", "creator", "creators", "byline", "owner", "publisher"}
_WHITESPACE_RE = re.compile(r"\s+")


def export_source_author_coverage_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write a deterministic per-source author coverage CSV."""
    unit_list = list(units)
    rows = _coverage_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "source_project_count": len(rows),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _coverage_rows(units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    groups: dict[str, list[KnowledgeUnit]] = defaultdict(list)
    for unit in units:
        groups[_unit_source(unit)].append(unit)

    rows: list[dict[str, str | int]] = []
    for source_project in sorted(groups, key=_sort_key):
        source_units = groups[source_project]
        author_counts: Counter[str] = Counter()
        units_with_author = 0
        for unit in source_units:
            authors = _unit_authors(unit)
            if authors:
                units_with_author += 1
            author_counts.update(authors)

        unit_count = len(source_units)
        rows.append(
            {
                "source_project": source_project,
                "unit_count": unit_count,
                "units_with_author": units_with_author,
                "author_coverage_percent": _decimal(units_with_author * 100 / unit_count),
                "distinct_authors": len(author_counts),
                "top_authors": _top_values(author_counts),
            }
        )
    return rows


def _unit_authors(unit: KnowledgeUnit) -> list[str]:
    metadata = unit.metadata if isinstance(unit.metadata, Mapping) else {}
    authors: set[str] = set()
    for key, value in metadata.items():
        if _inline_text(key).casefold() in _AUTHOR_KEYS:
            authors.update(_metadata_values(value))
    return sorted(authors, key=_sort_key)


def _metadata_values(value: object) -> list[str]:
    if isinstance(value, list | tuple | set):
        return sorted(
            {text for item in value for text in _metadata_values(item)},
            key=_sort_key,
        )
    text = _inline_text(value)
    return [text] if text else []


def _top_values(value_counts: Counter[str]) -> str:
    return "; ".join(
        f"{value} ({count})"
        for value, count in sorted(value_counts.items(), key=lambda item: (-item[1], _sort_key(item[0])))
    )


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _unit_source(unit: KnowledgeUnit) -> str:
    return _field_value(unit.source_project) or "Unknown"


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)


def _decimal(value: float) -> str:
    return f"{value:.2f}"
