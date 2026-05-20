"""CSV export for unit tax year transaction summaries."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

from graph.export._transaction_csv import CURRENCY_KEYS, UNKNOWN, decimal_text, first_amount, first_text, metadata, render_csv, sort_key, unit_date, write_output
from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["year", "currency", "transaction_count", "debit_total", "credit_total", "net_total", "first_date", "last_date"]


def export_units_to_tax_year_summary_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | Any | None = None,
) -> str | dict[str, Any]:
    """Return or write transaction summaries grouped by tax year and currency."""
    unit_list = list(units)
    rows = _summary_rows(unit_list)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    return write_output(path, text, {"unit_count": len(unit_list), "rows_exported": len(rows)})


def _summary_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], dict[str, Any]] = defaultdict(lambda: {"amounts": [], "dates": []})
    for unit in units:
        data = metadata(unit)
        amount = first_amount(data)
        txn_date = unit_date(unit)
        if amount is None or txn_date is None:
            continue
        currency = first_text(data, CURRENCY_KEYS) or UNKNOWN
        group = groups[(txn_date.year, currency)]
        group["amounts"].append(amount)
        group["dates"].append(txn_date)

    rows: list[dict[str, Any]] = []
    for (year, currency), group in groups.items():
        amounts = group["amounts"]
        dates = group["dates"]
        rows.append(
            {
                "year": str(year),
                "currency": currency,
                "transaction_count": len(amounts),
                "debit_total": decimal_text(sum((amount for amount in amounts if amount < 0), Decimal("0"))),
                "credit_total": decimal_text(sum((amount for amount in amounts if amount > 0), Decimal("0"))),
                "net_total": decimal_text(sum(amounts, Decimal("0"))),
                "first_date": min(dates).isoformat(),
                "last_date": max(dates).isoformat(),
            }
        )
    return sorted(rows, key=lambda row: (row["year"], sort_key(row["currency"])))
