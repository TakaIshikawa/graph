"""CSV export for unit amount summaries."""

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

_FIELDNAMES = [
    "source_project",
    "currency",
    "account",
    "category",
    "month",
    "unit_count",
    "total_amount",
    "debit_total",
    "credit_total",
    "min_amount",
    "max_amount",
    "representative_unit_ids",
]
_UNKNOWN = "Unknown"
_AMOUNT_KEYS = ("amount", "transaction_amount", "net_amount", "value", "total")
_CURRENCY_KEYS = ("currency",)
_ACCOUNT_KEYS = ("account", "account_name", "account_id", "card", "brokerage")
_CATEGORY_KEYS = ("category", "merchant_category", "type")
_DATE_KEYS = ("date", "source_date", "posted_at", "posted_date", "transaction_date", "month")
_DATE_FIELDS = ("created_at", "updated_at", "ingested_at")
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_amount_summary_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | Any | None = None,
) -> str | dict[str, Any]:
    """Return or write amount summaries grouped by source, currency, account, category, and month."""
    unit_list = list(units)
    rows = _summary_rows(unit_list)
    text = _render_csv(rows)
    if path is None:
        return text
    return _write_output(path, text, {"unit_count": len(unit_list), "rows_exported": len(rows)})


def _summary_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str, str, str, str], dict[str, Any]] = defaultdict(lambda: {"unit_ids": set(), "amounts": []})
    for unit in units:
        metadata = _metadata(unit)
        amount = _unit_amount(metadata)
        if amount is None:
            continue
        key = (
            _unit_source(unit),
            _first_text(metadata, _CURRENCY_KEYS) or _UNKNOWN,
            _first_text(metadata, _ACCOUNT_KEYS) or _UNKNOWN,
            _first_text(metadata, _CATEGORY_KEYS) or _UNKNOWN,
            _unit_month(unit) or _UNKNOWN,
        )
        groups[key]["amounts"].append(amount)
        if _unit_id(unit):
            groups[key]["unit_ids"].add(_unit_id(unit))

    rows: list[dict[str, str | int]] = []
    for (source, currency, account, category, month), group in groups.items():
        amounts = group["amounts"]
        rows.append(
            {
                "source_project": source,
                "currency": currency,
                "account": account,
                "category": category,
                "month": month,
                "unit_count": len(amounts),
                "total_amount": _decimal(sum(amounts, Decimal("0"))),
                "debit_total": _decimal(sum((amount for amount in amounts if amount < 0), Decimal("0"))),
                "credit_total": _decimal(sum((amount for amount in amounts if amount > 0), Decimal("0"))),
                "min_amount": _decimal(min(amounts)),
                "max_amount": _decimal(max(amounts)),
                "representative_unit_ids": _joined(group["unit_ids"]),
            }
        )
    return sorted(rows, key=lambda row: tuple(_sort_key(row[name]) for name in _FIELDNAMES[:5]))


def _unit_amount(metadata: Mapping[str, Any]) -> Decimal | None:
    for key in _AMOUNT_KEYS:
        if (amount := _amount_value(metadata.get(key))) is not None:
            return amount
    return None


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


def _unit_month(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    metadata = _metadata(unit)
    for key in _DATE_KEYS:
        text = _field_value(metadata.get(key))
        if re.fullmatch(r"\d{4}-\d{2}", text):
            return text
        if (parsed := _date_value(text)) is not None:
            return parsed.isoformat()[:7]
    for field in _DATE_FIELDS:
        if (parsed := _date_value(_get(unit, field))) is not None:
            return parsed.isoformat()[:7]
    return ""


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


def _first_text(metadata: Mapping[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        if text := _field_value(metadata.get(key)):
            return text
    return ""


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


def _decimal(value: Decimal) -> str:
    return format(value.normalize(), "f")
