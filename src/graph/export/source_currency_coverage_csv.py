"""CSV export for source currency coverage."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "source_project",
    "source_entity_type",
    "unit_count",
    "unique_currency_count",
    "currencies",
    "missing_currency_count",
    "dominant_currency",
    "representative_unit_ids",
]
_UNKNOWN = "Unknown"
_CURRENCY_KEYS = ("currency", "asset", "base_currency", "quote_currency", "transaction_currency", "coin")
_WHITESPACE_RE = re.compile(r"\s+")


def export_source_currency_coverage_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | Any | None = None,
) -> str | dict[str, Any]:
    """Return or write currency usage grouped by source project and entity type."""
    unit_list = list(units)
    rows = _coverage_rows(unit_list)
    text = _render_csv(rows)
    if path is None:
        return text
    return _write_output(path, text, {"unit_count": len(unit_list), "rows_exported": len(rows)})


def _coverage_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"unit_count": 0, "missing": 0, "currencies": Counter(), "ids": set()}
    )
    for unit in units:
        key = (_unit_source(unit), _unit_source_type(unit))
        group = groups[key]
        group["unit_count"] += 1
        if _unit_id(unit):
            group["ids"].add(_unit_id(unit))
        currencies = _unit_currencies(unit)
        if not currencies:
            group["missing"] += 1
        for currency in currencies:
            group["currencies"][currency] += 1

    rows: list[dict[str, str | int]] = []
    for (source, entity_type), group in groups.items():
        currencies = sorted(group["currencies"], key=_sort_key)
        dominant = ""
        if group["currencies"]:
            dominant = sorted(group["currencies"].items(), key=lambda item: (-item[1], _sort_key(item[0])))[0][0]
        rows.append(
            {
                "source_project": source,
                "source_entity_type": entity_type,
                "unit_count": group["unit_count"],
                "unique_currency_count": len(currencies),
                "currencies": "; ".join(currencies),
                "missing_currency_count": group["missing"],
                "dominant_currency": dominant,
                "representative_unit_ids": _joined(group["ids"]),
            }
        )
    return sorted(rows, key=lambda row: (_sort_key(row["source_project"]), _sort_key(row["source_entity_type"])))


def _unit_currencies(unit: KnowledgeUnit | Mapping[str, Any]) -> list[str]:
    metadata = _metadata(unit)
    values: list[str] = []
    for key in _CURRENCY_KEYS:
        value = metadata.get(key)
        if isinstance(value, (list, tuple, set)):
            values.extend(_currency_value(item) for item in value)
        else:
            values.append(_currency_value(value))
    return sorted({value for value in values if value}, key=_sort_key)


def _currency_value(value: object) -> str:
    text = _field_value(value)
    return text.upper() if len(text) <= 5 else text


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
    text = "" if value is None else str(getattr(value, "value", value))
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _field_value(value)
    return (text.casefold(), text)
