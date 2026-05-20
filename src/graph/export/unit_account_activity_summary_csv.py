"""CSV export for unit account activity summaries."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

from graph.export._transaction_csv import (
    ACCOUNT_KEYS,
    COUNTERPARTY_KEYS,
    CURRENCY_KEYS,
    UNKNOWN,
    decimal_text,
    first_amount,
    first_text,
    metadata,
    normalized_counterparty,
    render_csv,
    sort_key,
    unit_date,
    write_output,
)
from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "account",
    "currency",
    "transaction_count",
    "debit_total",
    "credit_total",
    "net_total",
    "first_activity_date",
    "last_activity_date",
    "distinct_counterparty_count",
]


def export_units_to_account_activity_summary_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | Any | None = None,
) -> str | dict[str, Any]:
    """Return or write transaction activity grouped by account and currency."""
    unit_list = list(units)
    rows = _summary_rows(unit_list)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    return write_output(path, text, {"unit_count": len(unit_list), "rows_exported": len(rows)})


def _summary_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"amounts": [], "dates": [], "counterparties": set()})
    for unit in units:
        data = metadata(unit)
        amount = first_amount(data)
        if amount is None:
            continue
        account = first_text(data, ACCOUNT_KEYS) or UNKNOWN
        currency = first_text(data, CURRENCY_KEYS) or UNKNOWN
        group = groups[(account, currency)]
        group["amounts"].append(amount)
        if (activity_date := unit_date(unit)) is not None:
            group["dates"].append(activity_date)
        if first_text(data, COUNTERPARTY_KEYS):
            group["counterparties"].add(normalized_counterparty(data))

    rows: list[dict[str, Any]] = []
    for (account, currency), group in groups.items():
        amounts = group["amounts"]
        dates = group["dates"]
        rows.append(
            {
                "account": account,
                "currency": currency,
                "transaction_count": len(amounts),
                "debit_total": decimal_text(sum((amount for amount in amounts if amount < 0), Decimal("0"))),
                "credit_total": decimal_text(sum((amount for amount in amounts if amount > 0), Decimal("0"))),
                "net_total": decimal_text(sum(amounts, Decimal("0"))),
                "first_activity_date": min(dates).isoformat() if dates else "",
                "last_activity_date": max(dates).isoformat() if dates else "",
                "distinct_counterparty_count": len(group["counterparties"]),
            }
        )
    return sorted(rows, key=lambda row: (sort_key(row["account"]), sort_key(row["currency"])))
