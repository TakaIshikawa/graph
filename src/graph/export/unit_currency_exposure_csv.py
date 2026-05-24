"""CSV export for unit currency exposure."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, flatten_values, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["currency", "unit_count", "amount_count", "total_amount", "unit_ids", "source_keys"]
_CURRENCY_KEYS = {"currency", "amount_currency", "transaction_currency", "price_currency"}
_AMOUNT_KEYS = {"amount", "price", "value", "total"}
_NESTED_KEYS = {"amount", "price", "transaction", "payment"}


def export_unit_currency_exposure_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write currency exposure grouped by normalized currency."""
    unit_list = list(units)
    rows = _rows(unit_list)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(units: list[Mapping[str, Any] | object]) -> list[dict[str, str | int]]:
    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"unit_ids": set(), "amounts": [], "source_keys": set()})
    for unit in units:
        for currency, amount, source_key in _currency_values(unit):
            bucket = groups[currency]
            if unit_id(unit):
                bucket["unit_ids"].add(unit_id(unit))
            if amount is not None:
                bucket["amounts"].append(amount)
            bucket["source_keys"].add(source_key)
    rows: list[dict[str, str | int]] = []
    for currency in sorted(groups, key=sort_key):
        bucket = groups[currency]
        rows.append(
            {
                "currency": currency,
                "unit_count": len(bucket["unit_ids"]),
                "amount_count": len(bucket["amounts"]),
                "total_amount": _decimal(sum(bucket["amounts"])) if bucket["amounts"] else "",
                "unit_ids": "; ".join(sorted(bucket["unit_ids"], key=sort_key)),
                "source_keys": "; ".join(sorted(bucket["source_keys"], key=sort_key)),
            }
        )
    return rows


def _currency_values(unit: Mapping[str, Any] | object) -> list[tuple[str, float | None, str]]:
    values: list[tuple[str, float | None, str]] = []
    data = metadata(unit)
    for key in _CURRENCY_KEYS:
        for item in flatten_values(get(unit, key)):
            currency = _normalize_currency(item)
            if currency:
                values.append((currency, _amount(unit), key))
        for item in flatten_values(data.get(key)):
            currency = _normalize_currency(item)
            if currency:
                values.append((currency, _amount(unit), f"metadata.{key}"))
    for key in _NESTED_KEYS:
        raw = data.get(key)
        items = raw if isinstance(raw, list | tuple | set) else [raw]
        for item in items:
            if isinstance(item, Mapping):
                values.extend(_nested_currency(item, f"metadata.{key}"))
    return values


def _nested_currency(values: Mapping[str, Any], prefix: str) -> list[tuple[str, float | None, str]]:
    rows: list[tuple[str, float | None, str]] = []
    amount = _amount_from_mapping(values)
    for key, value in values.items():
        if field_value(key).casefold().replace("-", "_").replace(" ", "_") in _CURRENCY_KEYS:
            currency = _normalize_currency(value)
            if currency:
                rows.append((currency, amount, f"{prefix}.{field_value(key)}"))
    return rows


def _amount(unit: Mapping[str, Any] | object) -> float | None:
    for key in _AMOUNT_KEYS:
        amount = _number(get(unit, key))
        if amount is not None:
            return amount
    return _amount_from_mapping(metadata(unit))


def _amount_from_mapping(values: Mapping[str, Any]) -> float | None:
    for key in _AMOUNT_KEYS:
        amount = _number(values.get(key))
        if amount is not None:
            return amount
    return None


def _normalize_currency(value: object) -> str:
    text = field_value(value)
    return text.upper() if 2 <= len(text) <= 5 and text.replace("_", "").isalpha() else text


def _number(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(field_value(value).replace(",", ""))
    except ValueError:
        return None


def _decimal(value: float) -> str:
    return f"{value:.2f}"
