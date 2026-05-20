"""CSV export for daily unit cashflow calendar summaries."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

from graph.export._transaction_csv import ACCOUNT_KEYS, CURRENCY_KEYS, UNKNOWN, decimal_text, first_amount, first_text, metadata, render_csv, sort_key, unit_date, unit_source, write_output
from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["date", "source_project", "account", "currency", "transaction_count", "debit_total", "credit_total", "net_amount", "running_balance_delta"]


def export_unit_cashflow_calendar_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | Any | None = None,
) -> str | dict[str, Any]:
    """Return or write daily transaction cashflow with running balance deltas."""
    unit_list = list(units)
    rows = _calendar_rows(unit_list)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    return write_output(path, text, {"unit_count": len(unit_list), "rows_exported": len(rows)})


def _calendar_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[Decimal]] = defaultdict(list)
    for unit in units:
        data = metadata(unit)
        amount = first_amount(data)
        txn_date = unit_date(unit)
        if amount is None or txn_date is None:
            continue
        key = (txn_date.isoformat(), unit_source(unit), first_text(data, ACCOUNT_KEYS) or UNKNOWN, first_text(data, CURRENCY_KEYS) or UNKNOWN)
        groups[key].append(amount)

    rows_by_series: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for (txn_date, source, account, currency), amounts in groups.items():
        rows_by_series[(source, account, currency)].append(
            {
                "date": txn_date,
                "source_project": source,
                "account": account,
                "currency": currency,
                "transaction_count": len(amounts),
                "debit_total": decimal_text(sum((amount for amount in amounts if amount < 0), Decimal("0"))),
                "credit_total": decimal_text(sum((amount for amount in amounts if amount > 0), Decimal("0"))),
                "net_amount": sum(amounts, Decimal("0")),
            }
        )

    rows: list[dict[str, Any]] = []
    for series_key in sorted(rows_by_series, key=lambda key: tuple(sort_key(part) for part in key)):
        running = Decimal("0")
        for row in sorted(rows_by_series[series_key], key=lambda item: item["date"]):
            running += row["net_amount"]
            row["net_amount"] = decimal_text(row["net_amount"])
            row["running_balance_delta"] = decimal_text(running)
            rows.append(row)
    return rows
