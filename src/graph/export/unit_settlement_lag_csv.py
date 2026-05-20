"""CSV export for unit transaction settlement lag summaries."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from decimal import Decimal
from pathlib import Path
from statistics import median
from typing import Any

from graph.export._transaction_csv import ACCOUNT_KEYS, CURRENCY_KEYS, UNKNOWN, date_value, decimal_text, first_text, joined, metadata, render_csv, sort_key, unit_id, unit_source, write_output
from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["source_project", "account", "currency", "count", "average_lag_days", "median_lag_days", "max_lag_days", "representative_unit_ids"]
_TRANSACTION_DATE_KEYS = ("trade_date", "transaction_date", "date", "source_date")
_SETTLEMENT_DATE_KEYS = ("settlement_date", "posted_date", "posted_at")


def export_unit_settlement_lag_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | Any | None = None,
) -> str | dict[str, Any]:
    """Return or write settlement lag summaries by source, account, and currency."""
    unit_list = list(units)
    rows = _lag_rows(unit_list)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    return write_output(path, text, {"unit_count": len(unit_list), "rows_exported": len(rows)})


def _lag_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(lambda: {"lags": [], "ids": set()})
    for unit in units:
        data = metadata(unit)
        start = _first_date(data, _TRANSACTION_DATE_KEYS)
        end = _first_date(data, _SETTLEMENT_DATE_KEYS)
        if start is None or end is None:
            continue
        key = (unit_source(unit), first_text(data, ACCOUNT_KEYS) or UNKNOWN, first_text(data, CURRENCY_KEYS) or UNKNOWN)
        group = groups[key]
        group["lags"].append((end - start).days)
        if unit_id(unit):
            group["ids"].add(unit_id(unit))

    rows: list[dict[str, Any]] = []
    for (source, account, currency), group in groups.items():
        lags = group["lags"]
        rows.append(
            {
                "source_project": source,
                "account": account,
                "currency": currency,
                "count": len(lags),
                "average_lag_days": decimal_text(sum(Decimal(lag) for lag in lags) / Decimal(len(lags))),
                "median_lag_days": decimal_text(Decimal(str(median(lags)))),
                "max_lag_days": max(lags),
                "representative_unit_ids": joined(group["ids"]),
            }
        )
    return sorted(rows, key=lambda row: tuple(sort_key(row[name]) for name in _FIELDNAMES[:3]))


def _first_date(metadata: Mapping[str, Any], keys: Iterable[str]):
    for key in keys:
        if (parsed := date_value(metadata.get(key))) is not None:
            return parsed
    return None
