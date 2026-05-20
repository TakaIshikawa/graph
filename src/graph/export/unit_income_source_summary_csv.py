"""CSV export for unit income source summaries."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

from graph.export._transaction_csv import (
    COUNTERPARTY_KEYS,
    CURRENCY_KEYS,
    DESCRIPTION_KEYS,
    UNKNOWN,
    decimal_text,
    field_value,
    first_amount,
    first_text,
    get_value,
    joined,
    metadata,
    normalized_text,
    render_csv,
    sort_key,
    unit_date,
    unit_id,
    write_output,
)
from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "source_label",
    "currency",
    "credit_count",
    "credit_total",
    "average_credit",
    "first_seen",
    "last_seen",
    "representative_unit_ids",
]
_INCOME_SOURCE_KEYS = ("income_source", "source", "source_label", "employer", "payer", "originator", *COUNTERPARTY_KEYS, *DESCRIPTION_KEYS)
_SOURCE_METADATA_FIELDS = ("source_metadata", "source", "provenance")


def export_units_to_income_source_summary_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | Any | None = None,
) -> str | dict[str, Any]:
    """Return or write positive transaction summaries grouped by income source and currency."""
    unit_list = list(units)
    rows = _summary_rows(unit_list)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    return write_output(path, text, {"unit_count": len(unit_list), "rows_exported": len(rows)})


def _summary_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"label": "", "amounts": [], "dates": [], "ids": set()})
    for unit in units:
        data = metadata(unit)
        amount = first_amount(data)
        if amount is None or amount <= 0:
            continue

        label = _income_source_label(unit, data)
        currency = first_text(data, CURRENCY_KEYS) or UNKNOWN
        group = groups[(normalized_text(label) or UNKNOWN.casefold(), currency)]
        group["label"] = group["label"] or label
        group["amounts"].append(amount)
        if (seen := unit_date(unit)) is not None:
            group["dates"].append(seen)
        if identifier := unit_id(unit):
            group["ids"].add(identifier)

    rows: list[dict[str, Any]] = []
    for (_, currency), group in groups.items():
        amounts = group["amounts"]
        dates = group["dates"]
        total = sum(amounts, Decimal("0"))
        rows.append(
            {
                "source_label": group["label"] or UNKNOWN,
                "currency": currency,
                "credit_count": len(amounts),
                "credit_total": decimal_text(total),
                "average_credit": decimal_text(total / Decimal(len(amounts))),
                "first_seen": min(dates).isoformat() if dates else "",
                "last_seen": max(dates).isoformat() if dates else "",
                "representative_unit_ids": joined(group["ids"]),
            }
        )
    return sorted(rows, key=lambda row: (sort_key(row["source_label"]), sort_key(row["currency"])))


def _income_source_label(unit: KnowledgeUnit | Mapping[str, Any], data: Mapping[str, Any]) -> str:
    if label := first_text(data, _INCOME_SOURCE_KEYS):
        return label
    for field in _SOURCE_METADATA_FIELDS:
        source_data = get_value(unit, field)
        if isinstance(source_data, Mapping):
            if label := first_text(source_data, _INCOME_SOURCE_KEYS):
                return label
    return field_value(get_value(unit, "source_project")) or UNKNOWN
