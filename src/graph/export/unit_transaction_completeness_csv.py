"""CSV export for transaction metadata completeness summaries."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

from graph.export._transaction_csv import ACCOUNT_KEYS, CATEGORY_KEYS, COUNTERPARTY_KEYS, CURRENCY_KEYS, DESCRIPTION_KEYS, UNKNOWN, decimal_text, first_amount, first_text, metadata, render_csv, sort_key, unit_date, unit_id, unit_source, write_output
from graph.types.models import KnowledgeUnit

_REQUIRED_FIELDS = ("amount", "currency", "date", "counterparty", "category", "description")
_FIELDNAMES = (
    ["source_project", "account", "total_count"]
    + [f"{field}_{suffix}" for field in _REQUIRED_FIELDS for suffix in ("present_count", "missing_count", "completeness_pct")]
    + ["sample_missing_unit_ids"]
)


def export_unit_transaction_completeness_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | Any | None = None,
) -> str | dict[str, Any]:
    """Return or write transaction metadata completeness by source and account."""
    unit_list = list(units)
    rows = _completeness_rows(unit_list)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    return write_output(path, text, {"unit_count": len(unit_list), "rows_exported": len(rows)})


def _completeness_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"total": 0, "present": defaultdict(int), "missing_ids": []})
    for unit in units:
        data = metadata(unit)
        if not _is_transaction_like(unit, data):
            continue
        key = (unit_source(unit), first_text(data, ACCOUNT_KEYS) or UNKNOWN)
        group = groups[key]
        group["total"] += 1
        present = _present_fields(unit, data)
        missing_any = False
        for field in _REQUIRED_FIELDS:
            if present[field]:
                group["present"][field] += 1
            else:
                missing_any = True
        if missing_any and unit_id(unit):
            group["missing_ids"].append(unit_id(unit))

    rows: list[dict[str, Any]] = []
    for (source, account), group in groups.items():
        total = group["total"]
        row: dict[str, Any] = {"source_project": source, "account": account, "total_count": total}
        for field in _REQUIRED_FIELDS:
            present_count = group["present"][field]
            missing_count = total - present_count
            row[f"{field}_present_count"] = present_count
            row[f"{field}_missing_count"] = missing_count
            row[f"{field}_completeness_pct"] = decimal_text((Decimal(present_count) / Decimal(total)) * Decimal("100")) if total else "0"
        row["sample_missing_unit_ids"] = "; ".join(sorted(set(group["missing_ids"]), key=sort_key)[:5])
        rows.append(row)
    return sorted(rows, key=lambda row: tuple(sort_key(row[name]) for name in _FIELDNAMES[:2]))


def _present_fields(unit: KnowledgeUnit | Mapping[str, Any], data: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "amount": first_amount(data) is not None,
        "currency": bool(first_text(data, CURRENCY_KEYS)),
        "date": unit_date(unit) is not None,
        "counterparty": bool(first_text(data, COUNTERPARTY_KEYS)),
        "category": bool(first_text(data, CATEGORY_KEYS)),
        "description": bool(first_text(data, DESCRIPTION_KEYS)),
    }


def _is_transaction_like(unit: KnowledgeUnit | Mapping[str, Any], data: Mapping[str, Any]) -> bool:
    entity_type = str(getattr(unit, "source_entity_type", "") if not isinstance(unit, Mapping) else unit.get("source_entity_type", ""))
    if "transaction" in entity_type.casefold():
        return True
    return any(_present_fields(unit, data).values())
