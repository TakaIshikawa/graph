"""CSV export for a deterministic unit review reading queue."""

from __future__ import annotations

import csv
import math
import re
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "unit_id",
    "title",
    "source_project",
    "source_entity_type",
    "priority_score",
    "reason_codes",
    "estimated_reading_minutes",
    "best_date",
]
_DATE_FIELDS = ("updated_at", "created_at", "ingested_at")
_DATE_METADATA_KEYS = ("published_at", "date", "source_date", "observed_at")
_READ_LATER_KEYS = ("read_later", "unread", "bookmarked", "saved", "pinned")
_WHITESPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"\b\w+\b")


def export_unit_reading_queue_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write a maintenance reading queue for review-worthy units."""
    unit_list = list(units)
    rows = _queue_rows(unit_list)
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


def _queue_rows(units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for unit in units:
        score, reasons = _score(unit)
        best_date = _best_date(unit)
        rows.append(
            {
                "unit_id": _field_value(unit.id),
                "title": _inline_text(unit.title),
                "source_project": _field_value(unit.source_project) or "Unknown",
                "source_entity_type": _field_value(unit.source_entity_type) or "Unknown",
                "priority_score": score,
                "reason_codes": ";".join(reasons),
                "estimated_reading_minutes": _estimated_minutes(unit.content),
                "best_date": best_date.isoformat() if best_date else "",
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -int(row["priority_score"]),
            _sort_key(row["best_date"]),
            _sort_key(row["unit_id"]),
        ),
    )


def _score(unit: KnowledgeUnit) -> tuple[int, list[str]]:
    metadata = unit.metadata if isinstance(unit.metadata, Mapping) else {}
    score = 0
    reasons: list[str] = []
    if any(_truthy(metadata.get(key)) for key in _READ_LATER_KEYS):
        score += 50
        reasons.append("read_later")
    word_count = len(_WORD_RE.findall(_inline_text(unit.content)))
    if word_count >= 800:
        score += 20
        reasons.append("long_content")
    elif 0 < word_count < 80:
        score += 5
        reasons.append("quick_read")
    confidence = _confidence(unit.confidence)
    if confidence is not None and confidence < 0.5:
        score += 15
        reasons.append("low_confidence")
    if _best_date(unit) is not None:
        score += 5
        reasons.append("dated")
    if not reasons:
        reasons.append("ordinary")
    return score, reasons


def _estimated_minutes(content: object) -> int:
    word_count = len(_WORD_RE.findall(_inline_text(content)))
    if word_count == 0:
        return 0
    return max(1, math.ceil(word_count / 200))


def _best_date(unit: KnowledgeUnit) -> date | None:
    metadata = unit.metadata if isinstance(unit.metadata, Mapping) else {}
    values = [_date_value(metadata.get(key)) for key in _DATE_METADATA_KEYS]
    values.extend(_date_value(getattr(unit, field, None)) for field in _DATE_FIELDS)
    dates = [value for value in values if value is not None]
    return max(dates) if dates else None


def _date_value(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _inline_text(value)
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        try:
            return date.fromisoformat(text)
        except ValueError:
            return None


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _inline_text(value).casefold() in {"1", "true", "yes", "y", "read_later", "unread"}


def _confidence(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


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
