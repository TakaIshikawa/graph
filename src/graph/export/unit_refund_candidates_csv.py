"""CSV export for likely refund transaction pairs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

from graph.export._transaction_csv import CURRENCY_KEYS, UNKNOWN, decimal_text, first_amount, first_text, metadata, normalized_counterparty, render_csv, sort_key, unit_date, unit_id, write_output
from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["debit_unit_id", "credit_unit_id", "merchant", "currency", "debit_amount", "credit_amount", "date_lag_days", "amount_delta", "confidence", "evidence"]


def export_unit_refund_candidates_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | Any | None = None,
    *,
    max_days: int = 30,
) -> str | dict[str, Any]:
    """Return or write likely purchase/refund pairs within ``max_days`` days."""
    unit_list = list(units)
    rows = _candidate_rows(unit_list, max_days=max_days)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    return write_output(path, text, {"unit_count": len(unit_list), "rows_exported": len(rows)})


def _candidate_rows(units: list[KnowledgeUnit | Mapping[str, Any]], *, max_days: int) -> list[dict[str, str | int]]:
    txns: list[dict[str, Any]] = []
    for unit in units:
        data = metadata(unit)
        amount = first_amount(data)
        txn_date = unit_date(unit)
        merchant = normalized_counterparty(data)
        if amount is None or amount == 0 or txn_date is None or merchant == UNKNOWN:
            continue
        txns.append({"id": unit_id(unit), "amount": amount, "currency": first_text(data, CURRENCY_KEYS) or UNKNOWN, "date": txn_date, "merchant": merchant})

    rows: list[dict[str, str | int]] = []
    for index, left in enumerate(txns):
        for right in txns[index + 1 :]:
            if left["currency"] != right["currency"] or left["merchant"] != right["merchant"]:
                continue
            if left["amount"] * right["amount"] >= 0:
                continue
            lag = abs((left["date"] - right["date"]).days)
            if lag > max_days:
                continue
            debit, credit = (left, right) if left["amount"] < 0 else (right, left)
            amount_delta = abs(abs(debit["amount"]) - abs(credit["amount"]))
            confidence = _confidence(lag, amount_delta, abs(debit["amount"]), max_days)
            rows.append(
                {
                    "debit_unit_id": debit["id"],
                    "credit_unit_id": credit["id"],
                    "merchant": debit["merchant"],
                    "currency": debit["currency"],
                    "debit_amount": decimal_text(debit["amount"]),
                    "credit_amount": decimal_text(credit["amount"]),
                    "date_lag_days": lag,
                    "amount_delta": decimal_text(amount_delta),
                    "confidence": decimal_text(confidence),
                    "evidence": f"opposite signs with same merchant and currency within {lag} days",
                }
            )
    return sorted(rows, key=lambda row: tuple(sort_key(row[name]) for name in _FIELDNAMES[:4]))


def _confidence(lag: int, amount_delta: Decimal, debit_abs: Decimal, max_days: int) -> Decimal:
    day_penalty = Decimal(lag) / Decimal(max(max_days, 1)) * Decimal("0.3")
    amount_penalty = (amount_delta / debit_abs * Decimal("0.5")) if debit_abs else Decimal("0")
    confidence = Decimal("1") - day_penalty - amount_penalty
    return max(Decimal("0"), confidence)
