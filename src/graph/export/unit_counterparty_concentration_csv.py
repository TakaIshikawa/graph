"""CSV export for unit counterparty concentration summaries."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

from graph.export._transaction_csv import ACCOUNT_KEYS, CURRENCY_KEYS, UNKNOWN, decimal_text, first_amount, first_text, metadata, normalized_counterparty, render_csv, sort_key, unit_date, unit_source, write_output
from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["source_project", "account", "currency", "counterparty", "total_amount_abs", "debit_total", "credit_total", "transaction_count", "share_of_source_amount", "first_seen", "last_seen"]


def export_unit_counterparty_concentration_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | Any | None = None,
) -> str | dict[str, Any]:
    """Return or write counterparty concentration by source, account, and currency."""
    unit_list = list(units)
    rows = _concentration_rows(unit_list)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    return write_output(path, text, {"unit_count": len(unit_list), "rows_exported": len(rows)})


def _concentration_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], dict[str, Any]] = defaultdict(lambda: {"amounts": [], "dates": []})
    totals: dict[tuple[str, str, str], Decimal] = defaultdict(lambda: Decimal("0"))
    for unit in units:
        data = metadata(unit)
        amount = first_amount(data)
        if amount is None:
            continue
        source = unit_source(unit)
        account = first_text(data, ACCOUNT_KEYS) or UNKNOWN
        currency = first_text(data, CURRENCY_KEYS) or UNKNOWN
        counterparty = normalized_counterparty(data)
        key = (source, account, currency, counterparty)
        groups[key]["amounts"].append(amount)
        if (seen := unit_date(unit)) is not None:
            groups[key]["dates"].append(seen)
        totals[(source, account, currency)] += abs(amount)

    rows: list[dict[str, Any]] = []
    for (source, account, currency, counterparty), group in groups.items():
        amounts = group["amounts"]
        dates = sorted(group["dates"])
        total_abs = sum((abs(amount) for amount in amounts), Decimal("0"))
        source_abs = totals[(source, account, currency)]
        rows.append(
            {
                "source_project": source,
                "account": account,
                "currency": currency,
                "counterparty": counterparty,
                "total_amount_abs": decimal_text(total_abs),
                "debit_total": decimal_text(sum((amount for amount in amounts if amount < 0), Decimal("0"))),
                "credit_total": decimal_text(sum((amount for amount in amounts if amount > 0), Decimal("0"))),
                "transaction_count": len(amounts),
                "share_of_source_amount": decimal_text(total_abs / source_abs) if source_abs else "0",
                "first_seen": dates[0].isoformat() if dates else "",
                "last_seen": dates[-1].isoformat() if dates else "",
            }
        )
    return sorted(rows, key=lambda row: tuple(sort_key(row[name]) for name in _FIELDNAMES[:4]))
