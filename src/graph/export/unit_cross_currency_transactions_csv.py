"""CSV export for transaction units with multiple currency fields."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._transaction_csv import (
    AMOUNT_KEYS,
    CURRENCY_KEYS,
    UNKNOWN,
    decimal_text,
    field_value,
    first_amount,
    first_text,
    metadata,
    render_csv,
    sort_key,
    unit_date,
    unit_id,
    unit_source,
    write_output,
)
from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["unit_id", "date", "amount", "currency", "alternate_currency", "alternate_amount", "exchange_rate", "source", "evidence"]
_CURRENCY_FIELD_PAIRS = [
    ("currency", AMOUNT_KEYS),
    ("transaction_currency", AMOUNT_KEYS),
    ("source_currency", ("source_amount", "original_amount", *AMOUNT_KEYS)),
    ("settlement_currency", ("settlement_amount", "settled_amount", "converted_amount")),
    ("fee_currency", ("fee_amount", "fee")),
    ("original_currency", ("original_amount", "source_amount", "foreign_amount")),
    ("converted_currency", ("converted_amount", "settlement_amount", "amount_converted")),
    ("foreign_currency", ("foreign_amount", "original_amount")),
]
_EXCHANGE_RATE_KEYS = ("exchange_rate", "fx_rate", "conversion_rate", "rate")


def export_units_to_cross_currency_transactions_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | Any | None = None,
) -> str | dict[str, Any]:
    """Return or write transactions that expose at least two distinct normalized currencies."""
    unit_list = list(units)
    rows = _transaction_rows(unit_list)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    return write_output(path, text, {"unit_count": len(unit_list), "rows_exported": len(rows)})


def _transaction_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for unit in units:
        data = metadata(unit)
        currency_fields = _currency_fields(data)
        normalized = {currency.casefold() for _, currency, _ in currency_fields}
        if len(normalized) < 2:
            continue

        primary_currency = first_text(data, CURRENCY_KEYS) or currency_fields[0][1]
        primary_amount = first_amount(data)
        txn_date = unit_date(unit)
        exchange_rate = first_text(data, _EXCHANGE_RATE_KEYS)
        source = unit_source(unit)
        for key, alternate_currency, alternate_amount in currency_fields:
            if alternate_currency.casefold() == primary_currency.casefold():
                continue
            rows.append(
                {
                    "unit_id": unit_id(unit),
                    "date": txn_date.isoformat() if txn_date else "",
                    "amount": decimal_text(primary_amount) if primary_amount is not None else "",
                    "currency": primary_currency or UNKNOWN,
                    "alternate_currency": alternate_currency,
                    "alternate_amount": alternate_amount,
                    "exchange_rate": exchange_rate,
                    "source": source,
                    "evidence": f"{key}={alternate_currency}",
                }
            )
    return sorted(rows, key=lambda row: (row["date"] == "", row["date"], sort_key(row["unit_id"]), sort_key(row["alternate_currency"])))


def _currency_fields(data: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    fields: list[tuple[str, str, str]] = []
    seen_keys: set[str] = set()
    for key, amount_keys in _CURRENCY_FIELD_PAIRS:
        currency = field_value(data.get(key))
        if not currency or key in seen_keys:
            continue
        seen_keys.add(key)
        fields.append((key, currency.upper(), _first_amount_text(data, amount_keys)))
    return fields


def _first_amount_text(data: Mapping[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        if key in data and (text := field_value(data.get(key))):
            return text
    return ""
