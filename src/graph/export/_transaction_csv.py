"""Small helpers for transaction-oriented CSV reports."""

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

UNKNOWN = "Unknown"
AMOUNT_KEYS = ("amount", "transaction_amount", "net_amount", "change", "value", "total")
ACCOUNT_KEYS = ("account", "account_name", "account_id", "card", "brokerage")
CATEGORY_KEYS = ("category", "merchant_category", "type")
COUNTERPARTY_KEYS = ("merchant", "merchant_name", "payee", "counterparty", "description", "name")
CURRENCY_KEYS = ("currency", "coin", "asset", "transaction_currency")
DATE_KEYS = ("date", "source_date", "posted_at", "posted_date", "transaction_date", "utc_timestamp", "UTC_Time")
DATE_FIELDS = ("created_at", "updated_at", "ingested_at")
DESCRIPTION_KEYS = ("description", "memo", "note", "details", "name")
_WHITESPACE_RE = re.compile(r"\s+")


def first_amount(metadata: Mapping[str, Any], keys: Iterable[str] = AMOUNT_KEYS) -> Decimal | None:
    for key in keys:
        if (amount := amount_value(metadata.get(key))) is not None:
            return amount
    return None


def amount_value(value: object) -> Decimal | None:
    text = field_value(value)
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


def unit_date(unit: KnowledgeUnit | Mapping[str, Any], keys: Iterable[str] = DATE_KEYS) -> date | None:
    data = metadata(unit)
    for key in keys:
        if (parsed := date_value(data.get(key))) is not None:
            return parsed
    for field in DATE_FIELDS:
        if (parsed := date_value(get_value(unit, field))) is not None:
            return parsed
    return None


def date_value(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = field_value(value)
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


def first_text(metadata: Mapping[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        if text := field_value(metadata.get(key)):
            return text
    return ""


def normalized_text(value: object) -> str:
    return field_value(value).casefold()


def normalized_counterparty(metadata: Mapping[str, Any]) -> str:
    return normalized_text(first_text(metadata, COUNTERPARTY_KEYS)) or UNKNOWN


def unit_id(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return field_value(get_value(unit, "id")) or field_value(get_value(unit, "source_id"))


def unit_source(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return field_value(get_value(unit, "source_project")) or UNKNOWN


def metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    data = get_value(unit, "metadata")
    return data if isinstance(data, Mapping) else {}


def get_value(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def joined(values: Iterable[object]) -> str:
    return "; ".join(sorted({field_value(value) for value in values if field_value(value)}, key=sort_key))


def render_csv(rows: list[Mapping[str, Any]], fieldnames: list[str]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def write_output(path: str | Path | Any, text: str, stats: dict[str, Any]) -> dict[str, Any]:
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


def field_value(value: object) -> str:
    text = "" if value is None else str(getattr(value, "value", value))
    return _WHITESPACE_RE.sub(" ", text).strip()


def sort_key(value: object) -> tuple[str, str]:
    text = field_value(value)
    return (text.casefold(), text)


def decimal_text(value: Decimal) -> str:
    return format(value.normalize(), "f")
