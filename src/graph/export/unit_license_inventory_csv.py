"""CSV export for unit license and rights metadata inventory."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "unit_id",
    "title",
    "source_project",
    "license",
    "license_family",
    "license_url",
    "rights_holder",
    "commercial_use",
    "derivatives_allowed",
]
_LICENSE_KEYS = ("license", "licence", "spdx_id", "rights", "usage_rights", "copyright", "copyright_status")
_LICENSE_URL_KEYS = ("license_url",)
_RIGHTS_HOLDER_KEYS = ("rights_holder",)
_COMMERCIAL_USE_KEYS = ("commercial_use",)
_DERIVATIVES_ALLOWED_KEYS = ("derivatives_allowed",)
_SPDX_ID_KEYS = ("spdx_id",)
_WHITESPACE_RE = re.compile(r"\s+")


def export_units_to_license_inventory_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write per-unit license and rights metadata inventory."""
    unit_list = list(units)
    rows = _inventory_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {"path": str(output_path), "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": output_path.stat().st_size}


def _inventory_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for unit in units:
        license_text = _lookup_text(unit, _LICENSE_KEYS)
        spdx_text = _lookup_text(unit, _SPDX_ID_KEYS)
        rows.append(
            {
                "unit_id": _unit_id(unit),
                "title": _field_value(_get(unit, "title")),
                "source_project": _field_value(_get(unit, "source_project")) or "Unknown",
                "license": license_text,
                "license_family": _license_family(license_text, spdx_text),
                "license_url": _lookup_text(unit, _LICENSE_URL_KEYS),
                "rights_holder": _lookup_text(unit, _RIGHTS_HOLDER_KEYS),
                "commercial_use": _lookup_text(unit, _COMMERCIAL_USE_KEYS),
                "derivatives_allowed": _lookup_text(unit, _DERIVATIVES_ALLOWED_KEYS),
            }
        )
    return sorted(rows, key=lambda row: (_sort_key(row["unit_id"]), _sort_key(row["title"])))


def _lookup_text(unit: KnowledgeUnit | Mapping[str, Any], keys: tuple[str, ...]) -> str:
    metadata = _metadata(unit)
    for key in keys:
        value = _get(unit, key)
        text = _field_value(value)
        if text:
            return text
        text = _joined_values(_casefold_get(metadata, key))
        if text:
            return text
    return ""


def _license_family(license_text: str, spdx_text: str) -> str:
    text = f"{license_text} {spdx_text}".casefold()
    if not text.strip():
        return "unknown"
    if spdx_text or re.search(r"\b(?:mit|apache-?2\.0|gpl-?3\.0|gpl-?2\.0|lgpl|agpl|bsd-?2|bsd-?3|mpl-?2\.0|isc)\b", text):
        return "spdx"
    if "public domain" in text or "cc0" in text or "unlicense" in text:
        return "public_domain"
    if "creative commons" in text or re.search(r"\bcc[- ]?(?:by|sa|nc|nd|zero)\b", text):
        return "creative_commons"
    if "copyright" in text or "all rights reserved" in text or "copyrighted" in text:
        return "copyrighted"
    return "custom"


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _casefold_get(mapping: Mapping[str, Any], key: str) -> object:
    for candidate_key, value in mapping.items():
        if _field_value(candidate_key).casefold() == key.casefold():
            return value
    return None


def _joined_values(value: object) -> str:
    values = sorted({_field_value(item) for item in _flatten(value) if _field_value(item)}, key=_sort_key)
    return "; ".join(values)


def _flatten(value: object) -> list[object]:
    if value is None or isinstance(value, bytes) or isinstance(value, Mapping):
        return []
    if isinstance(value, list | tuple | set):
        return [item for entry in value for item in _flatten(entry)]
    return [value]


def _render_csv(rows: list[dict[str, str]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _unit_id(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "id")) or _field_value(_get(unit, "source_id"))


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _field_value(value: object) -> str:
    if isinstance(value, list | tuple | set):
        return _joined_values(value)
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
