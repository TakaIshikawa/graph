"""CSV export for source-level link rot risk heuristics."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "source_project",
    "unit_count",
    "total_urls",
    "missing_url_count",
    "non_http_url_count",
    "archived_url_count",
    "stale_checked_count",
    "risk_level",
]
_URL_KEYS = ("url", "urls", "source_url", "external_url", "canonical_url", "link", "links")
_CHECKED_KEYS = ("last_seen", "last_checked", "checked_at", "url_checked_at", "link_checked_at")
_WHITESPACE_RE = re.compile(r"\s+")


def export_source_link_rot_risk_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
    *,
    reference_date: date | str | None = None,
    stale_after_days: int = 365,
) -> str | dict[str, Any]:
    """Return or write source-level link rot risk summaries without network calls."""
    unit_list = list(units)
    rows = _risk_rows(unit_list, _date_value(reference_date) or date.today(), stale_after_days)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {"path": str(output_path), "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": output_path.stat().st_size}


def _risk_rows(units: list[KnowledgeUnit | Mapping[str, Any]], reference_date: date, stale_after_days: int) -> list[dict[str, str | int]]:
    groups: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for unit in units:
        source = _field_value(_get(unit, "source_project")) or "Unknown"
        stats = groups[source]
        stats["unit_count"] += 1
        urls = _unit_urls(unit)
        if not urls:
            stats["missing_url_count"] += 1
        for url in urls:
            stats["total_urls"] += 1
            parsed = urlparse(url)
            scheme = parsed.scheme.casefold()
            host = parsed.netloc.casefold()
            if scheme not in {"http", "https"}:
                stats["non_http_url_count"] += 1
            if "web.archive.org" in host or "/web/" in parsed.path and "archive" in host:
                stats["archived_url_count"] += 1
        checked = _unit_checked_date(unit)
        if checked is None or (reference_date - checked).days > stale_after_days:
            stats["stale_checked_count"] += 1

    rows: list[dict[str, str | int]] = []
    for source, stats in groups.items():
        risk_points = stats["missing_url_count"] + stats["non_http_url_count"] + stats["stale_checked_count"]
        risk_level = "low" if risk_points == 0 else "medium" if risk_points <= max(1, stats["unit_count"]) else "high"
        rows.append(
            {
                "source_project": source,
                "unit_count": stats["unit_count"],
                "total_urls": stats["total_urls"],
                "missing_url_count": stats["missing_url_count"],
                "non_http_url_count": stats["non_http_url_count"],
                "archived_url_count": stats["archived_url_count"],
                "stale_checked_count": stats["stale_checked_count"],
                "risk_level": risk_level,
            }
        )
    return sorted(rows, key=lambda row: _sort_key(row["source_project"]))


def _unit_urls(unit: KnowledgeUnit | Mapping[str, Any]) -> list[str]:
    values: list[object] = []
    metadata = _metadata(unit)
    for key in _URL_KEYS:
        values.append(_get(unit, key))
        values.append(_casefold_get(metadata, key))
    return sorted({_field_value(value) for value in _flatten(values) if _field_value(value)}, key=_sort_key)


def _unit_checked_date(unit: KnowledgeUnit | Mapping[str, Any]) -> date | None:
    metadata = _metadata(unit)
    for key in _CHECKED_KEYS:
        parsed = _date_value(_casefold_get(metadata, key))
        if parsed is not None:
            return parsed
    return None


def _date_value(value: object) -> date | None:
    if value is None:
        return None
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
        return None


def _flatten(values: Iterable[object]) -> list[object]:
    flattened: list[object] = []
    for value in values:
        if isinstance(value, Mapping):
            continue
        if isinstance(value, list | tuple | set):
            flattened.extend(_flatten(value))
        else:
            flattened.append(value)
    return flattened


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _casefold_get(mapping: Mapping[str, Any], key: str) -> object:
    for candidate_key, value in mapping.items():
        if _field_value(candidate_key).casefold() == key.casefold():
            return value
    return None


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)

