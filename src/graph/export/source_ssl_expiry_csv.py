"""CSV export for source TLS/SSL certificate expiry hints."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, normalized_key, parse_datetime, render_csv, sort_key, source_id, write_csv

_FIELDNAMES = ["source_id", "url", "expiry_date", "days_until_expiry", "status", "source_key"]
_EXPIRY_KEYS = {"ssl_expires_at", "cert_expires_at", "certificate_expiry", "tls_not_after", "not_after"}
_NESTED_KEYS = {"url_metadata", "url", "fetch_metadata", "fetch", "request", "response", "tls", "ssl", "certificate"}
_URL_KEYS = ("url", "source_url", "canonical_url", "href")


def export_source_ssl_expiry_csv(
    sources: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
    reference_date: date | datetime | str | None = None,
    warning_days: int = 30,
) -> str | dict[str, Any]:
    """Return or write one row per source with TLS/SSL expiry status."""
    source_list = list(sources)
    reference = _reference_date(reference_date)
    rows = sorted((_row(source, reference, warning_days) for source in source_list), key=lambda row: sort_key(row["source_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "source_count": len(source_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(source: Mapping[str, Any] | object, reference: date, warning_days: int) -> dict[str, str]:
    raw, source_key = _expiry_value(source)
    parsed = _expiry_date(raw)
    if raw is None or field_value(raw) == "":
        expiry = ""
        days = ""
        status = "missing"
    elif parsed is None:
        expiry = field_value(raw)
        days = ""
        status = "invalid"
    else:
        delta = (parsed - reference).days
        expiry = parsed.isoformat()
        days = str(delta)
        status = "expired" if delta < 0 else "expiring_soon" if delta <= warning_days else "valid"
    return {
        "source_id": source_id(source),
        "url": _url(source),
        "expiry_date": expiry,
        "days_until_expiry": days,
        "status": status,
        "source_key": source_key,
    }


def _expiry_value(source: Mapping[str, Any] | object) -> tuple[object | None, str]:
    for key in _EXPIRY_KEYS:
        value = get(source, key)
        if field_value(value):
            return value, key
    found = _find_key(metadata(source), "metadata")
    if found[1]:
        return found
    for key in _NESTED_KEYS:
        value = get(source, key)
        if isinstance(value, Mapping):
            found = _find_key(value, key)
            if found[1]:
                return found
    return None, ""


def _find_key(values: Mapping[str, Any], prefix: str) -> tuple[object | None, str]:
    for key, value in values.items():
        key_text = field_value(key)
        if normalized_key(key_text) in _EXPIRY_KEYS and field_value(value):
            return value, f"{prefix}.{key_text}"
        if normalized_key(key_text) in _NESTED_KEYS and isinstance(value, Mapping):
            found = _find_key(value, f"{prefix}.{key_text}")
            if found[1]:
                return found
    return None, ""


def _expiry_date(value: object) -> date | None:
    parsed = parse_datetime(value)
    if parsed:
        return parsed.date()
    text = field_value(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _reference_date(value: date | datetime | str | None) -> date:
    if value is None:
        return datetime.now(timezone.utc).date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed = _expiry_date(value)
    if parsed is None:
        raise ValueError("reference_date must be an ISO date or datetime")
    return parsed


def _url(source: Mapping[str, Any] | object) -> str:
    data = metadata(source)
    for key in _URL_KEYS:
        text = field_value(get(source, key)) or field_value(data.get(key))
        if text:
            return text
    return ""
