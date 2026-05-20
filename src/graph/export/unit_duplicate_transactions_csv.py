"""CSV export for likely duplicate transaction pairs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._transaction_csv import ACCOUNT_KEYS, CURRENCY_KEYS, UNKNOWN, decimal_text, first_amount, first_text, metadata, normalized_counterparty, render_csv, sort_key, unit_date, unit_id, unit_source, write_output
from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["left_unit_id", "right_unit_id", "amount", "currency", "date", "counterparty", "same_source", "same_account", "evidence"]


def export_unit_duplicate_transactions_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | Any | None = None,
) -> str | dict[str, Any]:
    """Return or write likely duplicate transaction pairs."""
    unit_list = list(units)
    rows = _duplicate_rows(unit_list)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    return write_output(path, text, {"unit_count": len(unit_list), "rows_exported": len(rows)})


def _duplicate_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for unit in units:
        data = metadata(unit)
        amount = first_amount(data)
        txn_date = unit_date(unit)
        if amount is None or txn_date is None:
            continue
        currency = first_text(data, CURRENCY_KEYS) or UNKNOWN
        counterparty = normalized_counterparty(data)
        item = {
            "id": unit_id(unit),
            "amount": decimal_text(amount),
            "currency": currency,
            "date": txn_date.isoformat(),
            "counterparty": counterparty,
            "source": unit_source(unit),
            "account": first_text(data, ACCOUNT_KEYS) or UNKNOWN,
        }
        groups[(item["date"], item["amount"], currency, counterparty)].append(item)

    rows: list[dict[str, str]] = []
    for items in groups.values():
        ordered = sorted(items, key=lambda item: sort_key(item["id"]))
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                rows.append(
                    {
                        "left_unit_id": left["id"],
                        "right_unit_id": right["id"],
                        "amount": left["amount"],
                        "currency": left["currency"],
                        "date": left["date"],
                        "counterparty": left["counterparty"],
                        "same_source": _bool_text(left["source"] == right["source"]),
                        "same_account": _bool_text(left["account"] == right["account"]),
                        "evidence": "same date, amount, currency, and counterparty",
                    }
                )
    return sorted(rows, key=lambda row: tuple(sort_key(row[name]) for name in _FIELDNAMES[:6]))


def _bool_text(value: bool) -> str:
    return "true" if value else "false"
