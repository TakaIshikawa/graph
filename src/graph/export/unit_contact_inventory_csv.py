"""CSV export for unit contact metadata inventory."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import (
    field_value,
    flatten_values,
    get,
    metadata,
    render_csv,
    sort_key,
    unit_id,
    write_csv,
)

_FIELDNAMES = [
    "unit_id",
    "title",
    "emails",
    "phones",
    "people",
    "organizations",
    "handles",
    "contact_score",
]
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?:\+?\d[\d(). -]{7,}\d)")
_HANDLE_RE = re.compile(r"(?<!\w)@[A-Z0-9_]{2,30}\b", re.IGNORECASE)


def export_unit_contact_inventory_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write contact information detected per unit."""
    unit_list = list(units)
    rows = sorted((_row(unit) for unit in unit_list), key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {
        "path": output_path,
        "unit_count": len(unit_list),
        "rows_exported": len(rows),
        "bytes_written": bytes_written,
    }


def _row(unit: Mapping[str, Any] | object) -> dict[str, str | int]:
    data = metadata(unit)
    content = field_value(get(unit, "content"))
    emails = _unique(
        [*_values(data, "email", "emails", "contact_email"), *_EMAIL_RE.findall(content)]
    )
    phones = _unique(
        [
            *_values(data, "phone", "phones", "phone_number"),
            *(_clean_phone(value) for value in _PHONE_RE.findall(content)),
        ]
    )
    people = _unique(_values(data, "person", "people", "contact", "contacts", "author", "authors"))
    organizations = _unique(
        _values(data, "organization", "organizations", "org", "company", "institution")
    )
    handles = _unique(
        [
            *_values(data, "handle", "handles", "social", "social_handle"),
            *_HANDLE_RE.findall(content),
        ]
    )
    score = sum(1 for values in (emails, phones, people, organizations, handles) if values)
    return {
        "unit_id": unit_id(unit),
        "title": field_value(get(unit, "title")),
        "emails": ";".join(emails),
        "phones": ";".join(phones),
        "people": ";".join(people),
        "organizations": ";".join(organizations),
        "handles": ";".join(handles),
        "contact_score": score,
    }


def _values(data: Mapping[str, Any], *keys: str) -> list[str]:
    wanted = {key.casefold() for key in keys}
    values = []
    for key, value in data.items():
        if field_value(key).casefold() in wanted:
            values.extend(field_value(item) for item in flatten_values(value) if field_value(item))
    return values


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for value in values:
        text = field_value(value)
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def _clean_phone(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
