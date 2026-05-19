"""CSV export for unusually large transaction amounts."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation
from io import StringIO
from pathlib import Path
from statistics import median
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["unit_id", "source_project", "account", "category", "currency", "amount", "group_median_amount", "multiple_of_median", "threshold", "title"]
_UNKNOWN = "Unknown"
_AMOUNT_KEYS = ("amount", "transaction_amount", "net_amount", "change", "value", "total")
_ACCOUNT_KEYS = ("account", "account_name", "account_id", "card", "brokerage")
_CATEGORY_KEYS = ("category", "merchant_category", "type")
_CURRENCY_KEYS = ("currency", "coin", "asset", "transaction_currency")
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_amount_outliers_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | Any | None = None,
    *,
    minimum_group_size: int = 3,
    median_multiple_threshold: Decimal | int | str = Decimal("3"),
) -> str | dict[str, Any]:
    """Return or write units at least ``median_multiple_threshold`` times group median absolute amount."""
    unit_list = list(units)
    rows = _outlier_rows(unit_list, minimum_group_size=minimum_group_size, threshold=Decimal(str(median_multiple_threshold)))
    text = _render_csv(rows)
    if path is None:
        return text
    return _write_output(path, text, {"unit_count": len(unit_list), "rows_exported": len(rows)})


def _outlier_rows(units: list[KnowledgeUnit | Mapping[str, Any]], *, minimum_group_size: int, threshold: Decimal) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        metadata = _metadata(unit)
        amount = _first_amount(metadata, _AMOUNT_KEYS)
        if amount is None or amount == 0:
            continue
        key = (_unit_source(unit), _first_text(metadata, _ACCOUNT_KEYS) or _UNKNOWN, _first_text(metadata, _CATEGORY_KEYS) or _UNKNOWN, _first_text(metadata, _CURRENCY_KEYS) or _UNKNOWN)
        groups[key].append({"unit": unit, "amount": amount})
    rows: list[dict[str, str]] = []
    for (source, account, category, currency), items in groups.items():
        if len(items) < minimum_group_size:
            continue
        med = Decimal(str(median([abs(item["amount"]) for item in items])))
        if med == 0:
            continue
        for item in items:
            multiple = abs(item["amount"]) / med
            if multiple >= threshold:
                unit = item["unit"]
                rows.append({"unit_id": _unit_id(unit), "source_project": source, "account": account, "category": category, "currency": currency, "amount": _decimal(item["amount"]), "group_median_amount": _decimal(med), "multiple_of_median": _decimal(multiple), "threshold": _decimal(threshold), "title": _field_value(_get(unit, "title"))})
    return sorted(rows, key=lambda row: tuple(_sort_key(row[name]) for name in _FIELDNAMES[:5]))


def _first_amount(metadata: Mapping[str, Any], keys: Iterable[str]) -> Decimal | None:
    for key in keys:
        if (amount := _amount_value(metadata.get(key))) is not None:
            return amount
    return None


def _amount_value(value: object) -> Decimal | None:
    text = _field_value(value)
    if not text:
        return None
    negative = text.startswith("(") and text.endswith(")")
    cleaned = re.sub(r"[^0-9.+-]", "", text)
    if cleaned in {"", "+", "-", ".", "+.", "-."}:
        return None
    try:
        amount = Decimal(cleaned)
    except InvalidOperation:
        return None
    return -amount if negative else amount


def _first_text(metadata: Mapping[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        if text := _field_value(metadata.get(key)):
            return text
    return ""


def _unit_id(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "id")) or _field_value(_get(unit, "source_id"))


def _unit_source(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "source_project")) or _UNKNOWN


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _render_csv(rows: list[dict[str, str]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _write_output(path: str | Path | Any, text: str, stats: dict[str, Any]) -> dict[str, Any]:
    if hasattr(path, "write") and not isinstance(path, str | Path):
        written = path.write(text)
        stats["bytes_written"] = len(text.encode("utf-8")) if written is None else written
        return stats
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    stats["path"] = str(output_path)
    stats["bytes_written"] = output_path.stat().st_size
    return stats


def _field_value(value: object) -> str:
    text = "" if value is None else str(getattr(value, "value", value))
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _field_value(value)
    return (text.casefold(), text)


def _decimal(value: Decimal) -> str:
    return format(value.normalize(), "f")
