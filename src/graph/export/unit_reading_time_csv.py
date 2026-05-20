"""CSV export for estimated unit reading times."""

from __future__ import annotations

import csv
import math
import re
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["unit_id", "title", "source", "word_count", "estimated_minutes", "reading_speed_wpm"]
_WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")
DEFAULT_WORDS_PER_MINUTE = 200


def export_unit_reading_time_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
    *,
    words_per_minute: int = DEFAULT_WORDS_PER_MINUTE,
) -> str | dict[str, Any]:
    """Return or write per-unit reading time estimates from content word counts."""
    _validate_words_per_minute(words_per_minute)

    unit_list = list(units)
    rows = _reading_time_rows(unit_list, words_per_minute=words_per_minute)
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
        "reading_speed_wpm": words_per_minute,
        "bytes_written": output_path.stat().st_size,
    }


def _reading_time_rows(
    units: list[KnowledgeUnit | Mapping[str, Any]],
    *,
    words_per_minute: int,
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for unit in units:
        word_count = _word_count(_get(unit, "content"))
        rows.append(
            {
                "unit_id": _unit_id(unit),
                "title": _field_value(_get(unit, "title")),
                "source": _field_value(_get(unit, "source_project")) or "Unknown",
                "word_count": word_count,
                "estimated_minutes": _estimated_minutes(word_count, words_per_minute),
                "reading_speed_wpm": words_per_minute,
            }
        )
    return sorted(rows, key=lambda row: (_sort_key(row["unit_id"]), _sort_key(row["title"])))


def _estimated_minutes(word_count: int, words_per_minute: int) -> int:
    if word_count == 0:
        return 0
    return max(1, math.ceil(word_count / words_per_minute))


def _word_count(value: object) -> int:
    return len(_WORD_RE.findall(_field_value(value)))


def _validate_words_per_minute(value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("words_per_minute must be a positive integer")


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
