"""CSV export for sparse and dense unit content outliers."""

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
    "title_length",
    "content_char_count",
    "word_count",
    "bucket",
]
_WHITESPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"\b\w+\b")


def export_unit_content_length_outliers_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    include_normal: bool = False,
) -> str | dict[str, Any]:
    """Return or write non-normal unit length rows.

    Buckets are: empty (0 content chars), very_short (<20 words), short
    (<50 words), normal (<=2000 words), and long (>2000 words).
    """
    unit_list = list(units)
    rows = _outlier_rows(unit_list, include_normal=include_normal)
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


def _outlier_rows(units: list[KnowledgeUnit], *, include_normal: bool) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for unit in units:
        content = _inline_text(unit.content)
        title = _inline_text(unit.title)
        word_count = len(_WORD_RE.findall(content))
        bucket = _bucket(content_char_count=len(content), word_count=word_count)
        if bucket == "normal" and not include_normal:
            continue
        rows.append(
            {
                "unit_id": _field_value(unit.id),
                "title": title,
                "source_project": _field_value(unit.source_project) or "Unknown",
                "source_entity_type": _field_value(unit.source_entity_type) or "Unknown",
                "title_length": len(title),
                "content_char_count": len(content),
                "word_count": word_count,
                "bucket": bucket,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            _sort_key(row["source_project"]),
            _sort_key(row["source_entity_type"]),
            _sort_key(row["bucket"]),
            _sort_key(row["unit_id"]),
        ),
    )


def _bucket(*, content_char_count: int, word_count: int) -> str:
    if content_char_count == 0:
        return "empty"
    if word_count < 20:
        return "very_short"
    if word_count < 50:
        return "short"
    if word_count > 2000:
        return "long"
    return "normal"


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
