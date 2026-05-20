"""CSV export for large round transaction units."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

from graph.export._transaction_csv import (
    ACCOUNT_KEYS,
    COUNTERPARTY_KEYS,
    CURRENCY_KEYS,
    DESCRIPTION_KEYS,
    UNKNOWN,
    amount_value,
    decimal_text,
    first_amount,
    first_text,
    metadata,
    render_csv,
    sort_key,
    unit_date,
    unit_id,
    write_output,
)
from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["unit_id", "date", "amount", "currency", "account", "counterparty", "description", "round_increment"]


def export_units_to_large_round_transactions_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | Any | None = None,
    *,
    minimum_abs_amount: Decimal | int | str = Decimal("1000"),
    round_increment: Decimal | int | str = Decimal("100"),
) -> str | dict[str, Any]:
    """Return or write transactions with abs(amount) >= 1000 and divisible by 100 by default."""
    minimum = _positive_decimal(minimum_abs_amount, "minimum_abs_amount")
    increment = _positive_decimal(round_increment, "round_increment")
    unit_list = list(units)
    rows = _transaction_rows(unit_list, minimum, increment)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    return write_output(path, text, {"unit_count": len(unit_list), "rows_exported": len(rows)})


def _transaction_rows(units: list[KnowledgeUnit | Mapping[str, Any]], minimum_abs_amount: Decimal, round_increment: Decimal) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for unit in units:
        data = metadata(unit)
        amount = first_amount(data)
        if amount is None or abs(amount) < minimum_abs_amount or amount % round_increment != 0:
            continue
        txn_date = unit_date(unit)
        rows.append(
            {
                "unit_id": unit_id(unit),
                "date": txn_date.isoformat() if txn_date else "",
                "amount": decimal_text(amount),
                "currency": first_text(data, CURRENCY_KEYS) or UNKNOWN,
                "account": first_text(data, ACCOUNT_KEYS) or UNKNOWN,
                "counterparty": first_text(data, COUNTERPARTY_KEYS) or UNKNOWN,
                "description": first_text(data, DESCRIPTION_KEYS),
                "round_increment": decimal_text(round_increment),
            }
        )
    return sorted(rows, key=lambda row: (row["date"] == "", row["date"], -abs(amount_value(row["amount"]) or Decimal("0")), sort_key(row["unit_id"])))


def _positive_decimal(value: Decimal | int | str, name: str) -> Decimal:
    amount = amount_value(value)
    if amount is None or amount <= 0:
        raise ValueError(f"{name} must be a positive decimal value")
    return amount
