"""CSV export for unit review priority heuristics."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["unit_id", "title", "review_priority", "reasons", "missing_signal_count"]
_URL_RE = re.compile(r"\bhttps?://[^\s<>'\"]+")
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_review_priority_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
    *,
    reference_date: date | str | None = None,
    stale_after_days: int = 365,
) -> str | dict[str, Any]:
    """Return or write deterministic review priority scores for units."""
    unit_list = list(units)
    rows = _priority_rows(unit_list, _date_value(reference_date) or date.today(), stale_after_days)
    text = _render_csv(rows)
    if path is None:
        return text
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {"path": str(output_path), "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": output_path.stat().st_size}


def _priority_rows(units: list[KnowledgeUnit | Mapping[str, Any]], reference_date: date, stale_after_days: int) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for unit in units:
        reasons: list[str] = []
        if not _field_value(_get(unit, "content")):
            reasons.append("missing_content")
        if not (_field_value(_get(unit, "source_project")) or _field_value(_get(unit, "source_id"))):
            reasons.append("missing_source")
        if len(_tags(unit)) < 2:
            reasons.append("low_tag_count")
        updated = _date_value(_get(unit, "updated_at")) or _date_value(_casefold_get(_metadata(unit), "updated_at"))
        if updated is None or (reference_date - updated).days > stale_after_days:
            reasons.append("stale_timestamp")
        if _unresolved_link_count(unit):
            reasons.append("unresolved_links")
        rows.append(
            {
                "unit_id": _unit_id(unit),
                "title": _field_value(_get(unit, "title")),
                "review_priority": len(reasons),
                "reasons": "; ".join(reasons),
                "missing_signal_count": len(reasons),
            }
        )
    return sorted(rows, key=lambda row: (-int(row["review_priority"]), _sort_key(row["unit_id"])))


def _unresolved_link_count(unit: KnowledgeUnit | Mapping[str, Any]) -> int:
    text = _field_value(_get(unit, "content"))
    count = len(re.findall(r"\[\[[^\]]+\]\]", text))
    for url in _URL_RE.findall(text):
        if "example.invalid" in url or "TODO" in url.upper():
            count += 1
    return count


def _tags(unit: KnowledgeUnit | Mapping[str, Any]) -> list[str]:
    value = _get(unit, "tags")
    return [_field_value(item) for item in value if _field_value(item)] if isinstance(value, list | tuple | set) else []


def _date_value(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _field_value(value)
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        return None


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _casefold_get(mapping: Mapping[str, Any], key: str) -> object:
    for candidate_key, value in mapping.items():
        if _field_value(candidate_key).casefold() == key.casefold():
            return value
    return None


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
