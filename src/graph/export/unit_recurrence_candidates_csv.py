"""CSV export for likely recurring unit candidates."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["source_project", "recurrence_key", "unit_count", "first_seen", "last_seen", "span_days", "average_interval_days", "interval_bucket", "amount", "representative_unit_ids"]
_UNKNOWN = "Unknown"
_KEY_METADATA = ("merchant", "payee", "name", "title")
_AMOUNT_KEYS = ("amount", "transaction_amount", "net_amount")
_DATE_KEYS = ("date", "source_date", "posted_at", "posted_date", "transaction_date", "observed_at")
_DATE_FIELDS = ("created_at", "updated_at", "ingested_at")
_PUNCT_RE = re.compile(r"[^\w\s]+")
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_recurrence_candidates_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | Any | None = None,
) -> str | dict[str, Any]:
    """Return or write deterministic recurrence candidate rows."""
    unit_list = list(units)
    rows = _candidate_rows(unit_list)
    text = _render_csv(rows)
    if path is None:
        return text
    return _write_output(path, text, {"unit_count": len(unit_list), "rows_exported": len(rows)})


def _candidate_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(lambda: {"unit_ids": set(), "dates": []})
    for unit in units:
        unit_date = _unit_date(unit)
        if unit_date is None:
            continue
        key = (_unit_source(unit), _recurrence_key(unit), _unit_amount(_metadata(unit)))
        groups[key]["dates"].append(unit_date)
        if _unit_id(unit):
            groups[key]["unit_ids"].add(_unit_id(unit))

    rows: list[dict[str, str | int]] = []
    for (source, recurrence_key, amount), group in groups.items():
        dates = sorted(group["dates"])
        if len(dates) < 2:
            continue
        span_days = (dates[-1] - dates[0]).days
        average = span_days / (len(dates) - 1) if len(dates) > 1 else 0
        rows.append(
            {
                "source_project": source,
                "recurrence_key": recurrence_key,
                "unit_count": len(dates),
                "first_seen": dates[0].isoformat(),
                "last_seen": dates[-1].isoformat(),
                "span_days": span_days,
                "average_interval_days": f"{average:.2f}",
                "interval_bucket": _interval_bucket(average),
                "amount": amount,
                "representative_unit_ids": _joined(group["unit_ids"]),
            }
        )
    return sorted(rows, key=lambda row: (_sort_key(row["source_project"]), _sort_key(row["recurrence_key"]), _sort_key(row["first_seen"])))


def _recurrence_key(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    metadata = _metadata(unit)
    for key in _KEY_METADATA:
        if text := _field_value(metadata.get(key)):
            return _normalize(text)
    return _normalize(_field_value(_get(unit, "title"))) or _UNKNOWN


def _normalize(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", _PUNCT_RE.sub(" ", value.casefold())).strip()


def _interval_bucket(days: float) -> str:
    if days <= 2:
        return "daily"
    if days <= 10:
        return "weekly"
    if days <= 21:
        return "biweekly"
    if days <= 45:
        return "monthly"
    if days <= 120:
        return "quarterly"
    return "irregular"


def _unit_amount(metadata: Mapping[str, Any]) -> str:
    for key in _AMOUNT_KEYS:
        if (amount := _amount_value(metadata.get(key))) is not None:
            return format(amount.normalize(), "f")
    return ""


def _amount_value(value: object) -> Decimal | None:
    text = _field_value(value)
    if not text:
        return None
    negative = text.startswith("(") and text.endswith(")")
    cleaned = re.sub(r"[^0-9.+-]", "", text)
    if cleaned in {"", "+", "-", ".", "+.", "-."}:
        return None
    try:
        amount = Decimal(cleaned)
    except InvalidOperation:
        return None
    return -amount if negative else amount


def _unit_date(unit: KnowledgeUnit | Mapping[str, Any]) -> date | None:
    metadata = _metadata(unit)
    for key in _DATE_KEYS:
        if (parsed := _date_value(metadata.get(key))) is not None:
            return parsed
    for field in _DATE_FIELDS:
        if (parsed := _date_value(_get(unit, field))) is not None:
            return parsed
    return None


def _date_value(value: object) -> date | None:
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
        try:
            return date.fromisoformat(text)
        except ValueError:
            return None


def _unit_id(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "id")) or _field_value(_get(unit, "source_id"))


def _unit_source(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "source_project")) or _UNKNOWN


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _joined(values: Iterable[object]) -> str:
    return "; ".join(sorted({_field_value(value) for value in values if _field_value(value)}, key=_sort_key))


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _write_output(path: str | Path | Any, text: str, stats: dict[str, Any]) -> dict[str, Any]:
    if hasattr(path, "write") and not isinstance(path, str | Path):
        written = path.write(text)
        stats["bytes_written"] = len(text.encode("utf-8")) if written is None else written
        return stats
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    stats["path"] = str(output_path)
    stats["bytes_written"] = output_path.stat().st_size
    return stats


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
