"""CSV export for per-unit tag density diagnostics."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "unit_id",
    "title",
    "source_project",
    "source_entity_type",
    "tag_count",
    "unique_tag_count",
    "duplicate_tag_count",
    "normalized_tags",
    "content_word_count",
    "tags_per_100_words",
]
_WHITESPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"\b\w+\b")


def export_unit_tag_density_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write per-unit tag density rows."""
    unit_list = list(units)
    rows = _density_rows(unit_list)
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


def _density_rows(units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for unit in units:
        normalized_tags = _normalized_tags(getattr(unit, "tags", None))
        unique_tags = _unique_tags(normalized_tags)
        content_word_count = _word_count(getattr(unit, "content", ""))
        rows.append(
            {
                "unit_id": _field_value(unit.id),
                "title": _field_value(unit.title),
                "source_project": _field_value(unit.source_project) or "Unknown",
                "source_entity_type": _field_value(unit.source_entity_type) or "Unknown",
                "tag_count": len(normalized_tags),
                "unique_tag_count": len(unique_tags),
                "duplicate_tag_count": len(normalized_tags) - len(unique_tags),
                "normalized_tags": "; ".join(unique_tags),
                "content_word_count": content_word_count,
                "tags_per_100_words": _decimal(len(unique_tags) * 100 / content_word_count)
                if content_word_count
                else "0.00",
            }
        )

    return sorted(rows, key=lambda row: (_sort_key(row["unit_id"]), _sort_key(row["title"])))


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _normalized_tags(value: object) -> list[str]:
    if not isinstance(value, list | tuple | set):
        return []
    return [_inline_text(tag) for tag in value if _inline_text(tag)]


def _unique_tags(tags: list[str]) -> list[str]:
    by_folded: dict[str, str] = {}
    for tag in tags:
        folded = tag.casefold()
        if folded not in by_folded or _sort_key(tag) < _sort_key(by_folded[folded]):
            by_folded[folded] = tag
    return [tag for _, tag in sorted(by_folded.items(), key=lambda item: _sort_key(item[1]))]


def _word_count(value: object) -> int:
    return len(_WORD_RE.findall(_inline_text(value)))


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
