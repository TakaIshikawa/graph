"""CSV export for conflicting synonymous metadata values."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["unit_id", "source_project", "conflict_group", "keys_present", "normalized_values", "raw_values", "title"]
_GROUPS = {
    "date": ("date", "source_date", "observed_at", "published_at"),
    "url": ("url", "source_url", "permalink"),
    "title": ("title", "name", "description"),
    "amount": ("amount", "transaction_amount", "net_amount"),
    "account": ("account", "account_name", "account_id"),
}
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_metadata_conflicts_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | Any | None = None,
) -> str | dict[str, Any]:
    """Return or write rows for units with conflicting synonymous metadata values."""
    unit_list = list(units)
    rows = _conflict_rows(unit_list)
    text = _render_csv(rows)
    if path is None:
        return text
    return _write_output(path, text, {"unit_count": len(unit_list), "rows_exported": len(rows)})


def _conflict_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for unit in units:
        metadata = _metadata(unit)
        for group_name, keys in _GROUPS.items():
            present = [(key, metadata[key]) for key in keys if key in metadata and _field_value(metadata[key])]
            normalized = {_normalize(group_name, value) for _, value in present}
            normalized.discard("")
            if len(normalized) <= 1:
                continue
            rows.append(
                {
                    "unit_id": _unit_id(unit),
                    "source_project": _unit_source(unit),
                    "conflict_group": group_name,
                    "keys_present": _joined(key for key, _ in present),
                    "normalized_values": _joined(normalized),
                    "raw_values": _joined(f"{key}={_field_value(value)}" for key, value in present),
                    "title": _field_value(_get(unit, "title")),
                }
            )
    return sorted(rows, key=lambda row: (_sort_key(row["unit_id"]), _sort_key(row["conflict_group"])))


def _normalize(group_name: str, value: object) -> str:
    if group_name == "date":
        return _date_normalized(value) or _field_value(value)
    if group_name == "amount":
        amount = _amount_value(value)
        return format(amount.normalize(), "f") if amount is not None else _field_value(value)
    if group_name == "url":
        return _url_normalized(value)
    return _field_value(value).casefold()


def _date_normalized(value: object) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = _field_value(value)
    if not text:
        return ""
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text).date().isoformat()
    except ValueError:
        try:
            return date.fromisoformat(text).isoformat()
        except ValueError:
            return ""


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


def _url_normalized(value: object) -> str:
    text = _field_value(value)
    parts = urlsplit(text)
    if not parts.scheme or not parts.netloc:
        return text.casefold()
    path = parts.path.rstrip("/") or ""
    return urlunsplit((parts.scheme.casefold(), parts.netloc.casefold(), path, parts.query, ""))


def _unit_id(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "id")) or _field_value(_get(unit, "source_id"))


def _unit_source(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "source_project")) or "Unknown"


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _joined(values: Iterable[object]) -> str:
    return "; ".join(sorted({_field_value(value) for value in values if _field_value(value)}, key=_sort_key))


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
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
