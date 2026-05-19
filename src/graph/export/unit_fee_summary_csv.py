"""CSV export for unit fee summaries."""

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

_FIELDNAMES = ["source_project", "account", "currency", "fee_type", "month", "transaction_count", "fee_count", "total_fee", "average_fee", "max_fee", "representative_unit_ids"]
_UNKNOWN = "Unknown"
_FEE_KEYS = ("fee", "fees", "commission", "commission_amount", "network_fee", "transaction_fee")
_ACCOUNT_KEYS = ("account", "account_name", "account_id", "card", "brokerage")
_CURRENCY_KEYS = ("currency", "fee_currency", "coin", "asset", "transaction_currency")
_DATE_KEYS = ("date", "source_date", "posted_at", "posted_date", "transaction_date", "utc_timestamp", "UTC_Time", "month")
_DATE_FIELDS = ("created_at", "updated_at", "ingested_at")
_FEE_TYPES = {"fee": "Fee", "fees": "Fees", "commission": "Commission", "commission_amount": "Commission", "network_fee": "Network fee", "transaction_fee": "Transaction fee"}
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_fee_summary_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | Any | None = None,
) -> str | dict[str, Any]:
    """Return or write fee summaries grouped by source, account, currency, fee type, and month."""
    unit_list = list(units)
    rows = _summary_rows(unit_list)
    text = _render_csv(rows)
    if path is None:
        return text
    return _write_output(path, text, {"unit_count": len(unit_list), "rows_exported": len(rows)})


def _summary_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], dict[str, Any]] = defaultdict(lambda: {"fees": [], "unit_ids": set()})
    for unit in units:
        metadata = _metadata(unit)
        fees = _unit_fees(metadata)
        if not fees:
            continue
        for key, fee in fees:
            group_key = (_unit_source(unit), _first_text(metadata, _ACCOUNT_KEYS) or _UNKNOWN, _first_text(metadata, _CURRENCY_KEYS) or _UNKNOWN, _FEE_TYPES[key], _unit_month(unit) or _UNKNOWN)
            group = groups[group_key]
            group["fees"].append(fee)
            if _unit_id(unit):
                group["unit_ids"].add(_unit_id(unit))
    rows: list[dict[str, Any]] = []
    for (source, account, currency, fee_type, month), group in groups.items():
        fees = group["fees"]
        total = sum(fees, Decimal("0"))
        rows.append({"source_project": source, "account": account, "currency": currency, "fee_type": fee_type, "month": month, "transaction_count": len(group["unit_ids"]), "fee_count": len(fees), "total_fee": _decimal(total), "average_fee": _decimal(total / Decimal(len(fees))), "max_fee": _decimal(max(fees)), "representative_unit_ids": _joined(group["unit_ids"])})
    return sorted(rows, key=lambda row: tuple(_sort_key(row[name]) for name in _FIELDNAMES[:5]))


def _unit_fees(metadata: Mapping[str, Any]) -> list[tuple[str, Decimal]]:
    fees: list[tuple[str, Decimal]] = []
    for key in _FEE_KEYS:
        if (fee := _amount_value(metadata.get(key))) is not None:
            fees.append((key, fee))
    return fees


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
