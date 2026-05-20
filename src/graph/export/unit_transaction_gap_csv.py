"""CSV export for gaps between transaction units."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._transaction_csv import (
    ACCOUNT_KEYS,
    CURRENCY_KEYS,
    UNKNOWN,
    first_amount,
    first_text,
    joined,
    metadata,
    render_csv,
    sort_key,
    unit_date,
    unit_id,
    unit_source,
    write_output,
)
from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["account", "currency", "previous_date", "next_date", "gap_days", "previous_unit_id", "next_unit_id", "source"]


def export_units_to_transaction_gap_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | Any | None = None,
    *,
    minimum_gap_days: int = 30,
) -> str | dict[str, Any]:
    """Return or write account/currency transaction gaps longer than 30 days by default."""
    if minimum_gap_days < 0:
        raise ValueError("minimum_gap_days must be non-negative")
    unit_list = list(units)
    rows = _gap_rows(unit_list, minimum_gap_days)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    return write_output(path, text, {"unit_count": len(unit_list), "rows_exported": len(rows)})


def _gap_rows(units: list[KnowledgeUnit | Mapping[str, Any]], minimum_gap_days: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[tuple[Any, str, str]]] = defaultdict(list)
    for unit in units:
        data = metadata(unit)
        if first_amount(data) is None:
            continue
        txn_date = unit_date(unit)
        if txn_date is None:
            continue
        account = first_text(data, ACCOUNT_KEYS) or UNKNOWN
        currency = first_text(data, CURRENCY_KEYS) or UNKNOWN
        groups[(account, currency)].append((txn_date, unit_id(unit), unit_source(unit)))

    rows: list[dict[str, Any]] = []
    for (account, currency), entries in groups.items():
        ordered = sorted(entries, key=lambda entry: (entry[0], sort_key(entry[1])))
        for previous, current in zip(ordered, ordered[1:]):
            gap_days = (current[0] - previous[0]).days
            if gap_days <= minimum_gap_days:
                continue
            rows.append(
                {
                    "account": account,
                    "currency": currency,
                    "previous_date": previous[0].isoformat(),
                    "next_date": current[0].isoformat(),
                    "gap_days": gap_days,
                    "previous_unit_id": previous[1],
                    "next_unit_id": current[1],
                    "source": joined([previous[2], current[2]]),
                }
            )
    return sorted(rows, key=lambda row: (sort_key(row["account"]), sort_key(row["currency"]), row["previous_date"], row["next_date"], sort_key(row["previous_unit_id"])))
