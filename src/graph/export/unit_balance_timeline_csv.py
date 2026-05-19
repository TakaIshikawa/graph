"""CSV export for unit balance timelines."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["source_project", "account", "currency", "date", "balance", "amount", "delta_from_previous", "unit_id", "title"]
_UNKNOWN = "Unknown"
_BALANCE_KEYS = ("balance", "ending_balance", "running_balance", "current_balance", "available_balance")
_AMOUNT_KEYS = ("amount", "transaction_amount", "net_amount", "change", "value", "total")
_ACCOUNT_KEYS = ("account", "account_name", "account_id", "card", "brokerage")
_CURRENCY_KEYS = ("currency", "coin", "asset", "transaction_currency")
_DATE_KEYS = ("date", "source_date", "posted_at", "posted_date", "transaction_date", "utc_timestamp", "UTC_Time")
_DATE_FIELDS = ("created_at", "updated_at", "ingested_at")
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_balance_timeline_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | Any | None = None,
) -> str | dict[str, Any]:
    """Return or write one row per unit with parseable balance metadata."""
    unit_list = list(units)
    rows = _timeline_rows(unit_list)
    text = _render_csv(rows)
    if path is None:
        return text
    return _write_output(path, text, {"unit_count": len(unit_list), "rows_exported": len(rows)})


def _timeline_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for unit in units:
        metadata = _metadata(unit)
        balance = _first_amount(metadata, _BALANCE_KEYS)
        if balance is None:
            continue
        rows.append(
            {
                "source_project": _unit_source(unit),
                "account": _first_text(metadata, _ACCOUNT_KEYS) or _UNKNOWN,
                "currency": _first_text(metadata, _CURRENCY_KEYS) or _UNKNOWN,
                "date": _unit_date(unit),
                "balance": balance,
                "amount": _first_amount(metadata, _AMOUNT_KEYS),
                "delta_from_previous": "",
                "unit_id": _unit_id(unit),
                "title": _field_value(_get(unit, "title")),
            }
        )
    rows.sort(key=lambda row: (_sort_key(row["source_project"]), _sort_key(row["account"]), _sort_key(row["currency"]), _sort_key(row["date"]), _sort_key(row["unit_id"])))
    previous: dict[tuple[str, str, str], Decimal] = {}
    for row in rows:
        key = (row["source_project"], row["account"], row["currency"])
        if key in previous:
            row["delta_from_previous"] = _decimal(row["balance"] - previous[key])
        previous[key] = row["balance"]
        row["balance"] = _decimal(row["balance"])
        row["amount"] = _decimal(row["amount"]) if row["amount"] is not None else ""
    return rows


def _first_amount(metadata: Mapping[str, Any], keys: Iterable[str]) -> Decimal | None:
    for key in keys:
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


def _unit_date(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    metadata = _metadata(unit)
    for key in _DATE_KEYS:
        if (parsed := _date_value(metadata.get(key))) is not None:
            return parsed.isoformat()
    for field in _DATE_FIELDS:
        if (parsed := _date_value(_get(unit, field))) is not None:
            return parsed.isoformat()
    return _UNKNOWN


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


def _render_csv(rows: list[dict[str, Any]]) -> str:
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
    text = "" if value is None else str(getattr(value, "value", value))
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _field_value(value)
    return (text.casefold(), text)


def _decimal(value: Decimal) -> str:
    return format(value.normalize(), "f")
