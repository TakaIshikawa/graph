"""CSV export for temporal evidence spans on relations."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Any

_FIELDNAMES = [
    "relation_type",
    "source_id",
    "target_id",
    "evidence_count",
    "first_evidence_date",
    "last_evidence_date",
    "evidence_span_days",
    "has_multi_date_span",
]
_SOURCE_KEYS = ("source_id", "from_unit_id", "source_unit_id", "from_id", "source")
_TARGET_KEYS = ("target_id", "to_unit_id", "target_unit_id", "to_id", "target")
_RELATION_KEYS = ("relation_type", "relation", "type", "label")
_DATE_KEYS = (
    "evidence_date",
    "evidence_at",
    "date",
    "observed_at",
    "observed_date",
    "occurred_at",
    "timestamp",
    "published_at",
)
_EVIDENCE_KEYS = ("evidence", "evidence_items", "evidence_dates", "observations", "citations")
_METADATA_KEYS = ("metadata", "attributes", "attrs")
_WHITESPACE_RE = re.compile(r"\s+")


def export_relation_evidence_span_csv(
    edges: Iterable[Any],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write evidence date spans for relation-like edge records."""
    edge_list = list(edges)
    rows = _span_rows(edge_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "relation_count": len(edge_list),
        "dated_relation_count": sum(1 for row in rows if row["evidence_count"]),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _span_rows(edges: list[Any]) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for edge in edges:
        dates = sorted(_evidence_dates(edge))
        first = dates[0] if dates else None
        last = dates[-1] if dates else None
        span_days = (last - first).days if first and last else None
        rows.append(
            {
                "relation_type": _first_text(edge, _RELATION_KEYS),
                "source_id": _first_text(edge, _SOURCE_KEYS),
                "target_id": _first_text(edge, _TARGET_KEYS),
                "evidence_count": len(dates),
                "first_evidence_date": first.isoformat() if first else "",
                "last_evidence_date": last.isoformat() if last else "",
                "evidence_span_days": span_days if span_days is not None else "",
                "has_multi_date_span": "true" if span_days and span_days > 0 else "false",
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            _sort_key(row["relation_type"]),
            _sort_key(row["source_id"]),
            _sort_key(row["target_id"]),
        ),
    )


def _evidence_dates(edge: Any) -> list[date]:
    values: list[Any] = []
    values.extend(_values_for_keys(edge, _DATE_KEYS))
    for metadata_key in _METADATA_KEYS:
        metadata = _value(edge, metadata_key)
        if isinstance(metadata, Mapping):
            values.extend(_values_for_keys(metadata, _DATE_KEYS))
            values.extend(_evidence_item_values(metadata))
    values.extend(_evidence_item_values(edge))
    return [parsed for value in values if (parsed := _date_value(value)) is not None]


def _evidence_item_values(container: Any) -> list[Any]:
    values: list[Any] = []
    for key in _EVIDENCE_KEYS:
        evidence = _value(container, key)
        if evidence is None:
            continue
        if isinstance(evidence, list | tuple | set):
            items = evidence
        else:
            items = [evidence]
        for item in items:
            if isinstance(item, Mapping):
                values.extend(_values_for_keys(item, _DATE_KEYS))
                metadata = item.get("metadata")
                if isinstance(metadata, Mapping):
                    values.extend(_values_for_keys(metadata, _DATE_KEYS))
            else:
                values.append(item)
    return values


def _values_for_keys(container: Any, keys: tuple[str, ...]) -> list[Any]:
    return [_value(container, key) for key in keys if _value(container, key) is not None]


def _first_text(container: Any, keys: tuple[str, ...]) -> str:
    for key in keys:
        text = _field_value(_value(container, key))
        if text:
            return text
    return ""


def _value(container: Any, key: str) -> Any:
    if isinstance(container, Mapping):
        return container.get(key)
    return getattr(container, key, None)


def _date_value(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _inline_text(value)
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


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
