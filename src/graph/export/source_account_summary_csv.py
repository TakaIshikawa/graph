"""CSV export for account-oriented source summaries."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "source_project",
    "institution",
    "account",
    "source_entity_type",
    "unit_count",
    "first_seen",
    "last_seen",
    "currency",
    "total_amount",
    "representative_unit_ids",
]
_UNKNOWN = "Unknown"
_ACCOUNT_KEYS = ("account", "account_name", "account_id", "brokerage", "card")
_INSTITUTION_KEYS = ("institution", "brokerage")
_CURRENCY_KEYS = ("currency",)
_AMOUNT_KEYS = ("amount", "transaction_amount", "net_amount", "value", "total")
_DATE_FIELDS = ("created_at", "updated_at", "ingested_at")
_DATE_KEYS = ("date", "source_date", "posted_at", "posted_date", "transaction_date", "observed_at", "created_at", "updated_at")
_WHITESPACE_RE = re.compile(r"\s+")


def export_source_account_summary_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | Any | None = None,
) -> str | dict[str, Any]:
    """Return or write account summaries grouped by source, institution, account, and type."""
    unit_list = list(units)
    rows = _summary_rows(unit_list)
    text = _render_csv(rows)
    if path is None:
        return text
    return _write_output(path, text, {"unit_count": len(unit_list), "rows_exported": len(rows)})


def _summary_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str, str, str, str], dict[str, Any]] = defaultdict(
        lambda: {"unit_ids": set(), "dates": [], "amounts": []}
    )
    for unit in units:
        metadata = _metadata(unit)
        currency = _first_text(metadata, _CURRENCY_KEYS)
        key = (
            _unit_source(unit),
            _first_text(metadata, _INSTITUTION_KEYS) or _UNKNOWN,
            _first_text(metadata, _ACCOUNT_KEYS) or _UNKNOWN,
            _unit_source_type(unit),
            currency,
        )
        group = groups[key]
        if _unit_id(unit):
            group["unit_ids"].add(_unit_id(unit))
        if (unit_date := _unit_date(unit)) is not None:
            group["dates"].append(unit_date)
        if (amount := _unit_amount(metadata)) is not None:
            group["amounts"].append(amount)

    rows: list[dict[str, str | int]] = []
    for (source, institution, account, entity_type, currency), group in groups.items():
        dates = sorted(group["dates"])
        rows.append(
            {
                "source_project": source,
                "institution": institution,
                "account": account,
                "source_entity_type": entity_type,
                "unit_count": len(group["unit_ids"]),
                "first_seen": dates[0].isoformat() if dates else "",
                "last_seen": dates[-1].isoformat() if dates else "",
                "currency": currency,
                "total_amount": _decimal(sum(group["amounts"], Decimal("0"))) if group["amounts"] else "",
                "representative_unit_ids": _joined(group["unit_ids"]),
            }
        )
    return sorted(rows, key=lambda row: tuple(_sort_key(row[name]) for name in _FIELDNAMES[:4]))


def _unit_date(unit: KnowledgeUnit | Mapping[str, Any]) -> date | None:
    metadata = _metadata(unit)
    for key in _DATE_KEYS:
        if (parsed := _date_value(metadata.get(key))) is not None:
            return parsed
    for field in _DATE_FIELDS:
        if (parsed := _date_value(_get(unit, field))) is not None:
            return parsed
    return None


def _date_value(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _field_value(value)
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        try:
            return date.fromisoformat(text)
        except ValueError:
            return None


def _unit_amount(metadata: Mapping[str, Any]) -> Decimal | None:
    for key in _AMOUNT_KEYS:
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


def _unit_source_type(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "source_entity_type")) or _UNKNOWN


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _joined(values: Iterable[object]) -> str:
    return "; ".join(sorted({_field_value(value) for value in values if _field_value(value)}, key=_sort_key))


def _render_csv(rows: list[dict[str, str | int]]) -> str:
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
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)


def _decimal(value: Decimal) -> str:
    return format(value.normalize(), "f")
