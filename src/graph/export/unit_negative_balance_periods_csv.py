"""CSV export for contiguous unit negative balance periods."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from graph.export._transaction_csv import ACCOUNT_KEYS, CURRENCY_KEYS, decimal_text, first_amount, first_text, joined, metadata, render_csv, sort_key, unit_date, unit_id, write_output
from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["account", "currency", "start_date", "end_date", "days", "minimum_balance", "ending_balance", "sample_unit_ids"]
_BALANCE_KEYS = ("balance", "ending_balance", "running_balance", "current_balance", "available_balance")


@dataclass
class _BalancePoint:
    account: str
    currency: str
    date: date
    balance: Decimal
    unit_id: str


@dataclass
class _NegativePeriod:
    start_date: date
    end_date: date
    minimum_balance: Decimal
    ending_balance: Decimal
    unit_ids: list[str] = field(default_factory=list)


def export_units_to_negative_balance_periods_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | Any | None = None,
) -> str | dict[str, Any]:
    """Return or write contiguous periods where account balances are below zero."""
    unit_list = list(units)
    rows = _period_rows(unit_list)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    return write_output(path, text, {"unit_count": len(unit_list), "rows_exported": len(rows)})


def _period_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, Any]]:
    points_by_series: dict[tuple[str, str], list[_BalancePoint]] = defaultdict(list)
    for unit in units:
        data = metadata(unit)
        account = first_text(data, ACCOUNT_KEYS)
        currency = first_text(data, CURRENCY_KEYS)
        balance = first_amount(data, _BALANCE_KEYS)
        balance_date = unit_date(unit)
        if not account or not currency or balance is None or balance_date is None:
            continue
        points_by_series[(account, currency)].append(_BalancePoint(account=account, currency=currency, date=balance_date, balance=balance, unit_id=unit_id(unit)))

    rows: list[dict[str, Any]] = []
    for account, currency in sorted(points_by_series, key=lambda key: tuple(sort_key(part) for part in key)):
        active: _NegativePeriod | None = None
        for point in sorted(points_by_series[(account, currency)], key=lambda item: (item.date, sort_key(item.unit_id))):
            if point.balance < 0:
                if active is None:
                    active = _NegativePeriod(start_date=point.date, end_date=point.date, minimum_balance=point.balance, ending_balance=point.balance)
                else:
                    active.end_date = point.date
                    active.minimum_balance = min(active.minimum_balance, point.balance)
                    active.ending_balance = point.balance
                if point.unit_id:
                    active.unit_ids.append(point.unit_id)
                continue

            if active is not None:
                rows.append(_row(account, currency, active))
                active = None
        if active is not None:
            rows.append(_row(account, currency, active))
    return rows


def _row(account: str, currency: str, period: _NegativePeriod) -> dict[str, Any]:
    return {
        "account": account,
        "currency": currency,
        "start_date": period.start_date.isoformat(),
        "end_date": period.end_date.isoformat(),
        "days": (period.end_date - period.start_date).days + 1,
        "minimum_balance": decimal_text(period.minimum_balance),
        "ending_balance": decimal_text(period.ending_balance),
        "sample_unit_ids": joined(period.unit_ids),
    }
