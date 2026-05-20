"""CSV export for transaction activity by weekday."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

from graph.export._transaction_csv import ACCOUNT_KEYS, CURRENCY_KEYS, UNKNOWN, decimal_text, first_amount, first_text, metadata, render_csv, sort_key, unit_date, unit_source, write_output
from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["source_project", "account", "currency", "weekday_number", "weekday_name", "transaction_count", "debit_total", "credit_total", "net_amount", "average_abs_amount"]


def export_unit_weekday_activity_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | Any | None = None,
) -> str | dict[str, Any]:
    """Return or write transaction activity grouped by source, account, currency, and weekday."""
    unit_list = list(units)
    rows = _activity_rows(unit_list)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    return write_output(path, text, {"unit_count": len(unit_list), "rows_exported": len(rows)})


def _activity_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, int], list[Decimal]] = defaultdict(list)
    weekday_names: dict[tuple[str, str, str, int], str] = {}
    for unit in units:
        data = metadata(unit)
        amount = first_amount(data)
        txn_date = unit_date(unit)
        if amount is None or txn_date is None:
            continue
        weekday_number = txn_date.weekday() + 1
        key = (unit_source(unit), first_text(data, ACCOUNT_KEYS) or UNKNOWN, first_text(data, CURRENCY_KEYS) or UNKNOWN, weekday_number)
        groups[key].append(amount)
        weekday_names[key] = txn_date.strftime("%A")

    rows: list[dict[str, Any]] = []
    for (source, account, currency, weekday_number), amounts in groups.items():
        rows.append(
            {
                "source_project": source,
                "account": account,
                "currency": currency,
                "weekday_number": weekday_number,
                "weekday_name": weekday_names[(source, account, currency, weekday_number)],
                "transaction_count": len(amounts),
                "debit_total": decimal_text(sum((amount for amount in amounts if amount < 0), Decimal("0"))),
                "credit_total": decimal_text(sum((amount for amount in amounts if amount > 0), Decimal("0"))),
                "net_amount": decimal_text(sum(amounts, Decimal("0"))),
                "average_abs_amount": decimal_text(sum((abs(amount) for amount in amounts), Decimal("0")) / Decimal(len(amounts))),
            }
        )
    return sorted(rows, key=lambda row: (*[sort_key(row[name]) for name in _FIELDNAMES[:3]], row["weekday_number"]))
