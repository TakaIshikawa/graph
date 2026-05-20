"""CSV export for URL scheme inventory across units."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["scheme", "source_project", "source_entity_type", "unit_count", "url_count", "sample_units"]
_URL_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9+.-]*:[^\s<>'\"]+")
_WHITESPACE_RE = re.compile(r"\s+")
_SAMPLE_LIMIT = 3


def export_unit_url_scheme_inventory_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write counts of URL schemes by unit source and entity type."""
    unit_list = list(units)
    rows = _inventory_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "scheme_group_count": len(rows),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _inventory_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(lambda: {"unit_ids": set(), "url_count": 0})
    for unit in units:
        source_project = _field_value(_get(unit, "source_project")) or "Unknown"
        entity_type = _field_value(_get(unit, "source_entity_type")) or "Unknown"
        unit_id = _unit_id(unit)
        for scheme in _unit_schemes(unit):
            key = (scheme, source_project, entity_type)
            groups[key]["unit_ids"].add(unit_id)
            groups[key]["url_count"] += 1

    rows: list[dict[str, str | int]] = []
    for (scheme, source_project, entity_type), group in sorted(
        groups.items(), key=lambda item: (_sort_key(item[0][0]), _sort_key(item[0][1]), _sort_key(item[0][2]))
    ):
        sample_units = sorted(group["unit_ids"], key=_sort_key)[:_SAMPLE_LIMIT]
        rows.append(
            {
                "scheme": scheme,
                "source_project": source_project,
                "source_entity_type": entity_type,
                "unit_count": len(group["unit_ids"]),
                "url_count": group["url_count"],
                "sample_units": "; ".join(sample_units),
            }
        )
    return rows


def _unit_schemes(unit: KnowledgeUnit | Mapping[str, Any]) -> list[str]:
    values = [_get(unit, "content")]
    metadata = _get(unit, "metadata")
    if isinstance(metadata, Mapping):
        values.extend(_metadata_values(metadata))
    schemes: list[str] = []
    for value in values:
        schemes.extend(_schemes_from_text(value))
    return schemes


def _metadata_values(value: object) -> list[object]:
    if isinstance(value, Mapping):
        return [item for child in value.values() for item in _metadata_values(child)]
    if isinstance(value, list | tuple | set):
        return [item for child in value for item in _metadata_values(child)]
    return [value]


def _schemes_from_text(value: object) -> list[str]:
    if value is None or isinstance(value, bytes):
        return []
    schemes: list[str] = []
    for candidate in _URL_RE.findall(str(value)):
        parsed = urlparse(candidate.rstrip(".,);]"))
        if parsed.scheme and _looks_like_url(parsed.scheme, candidate):
            schemes.append(parsed.scheme.casefold())
    return schemes


def _looks_like_url(scheme: str, candidate: str) -> bool:
    if "://" in candidate:
        parsed = urlparse(candidate)
        return bool(parsed.netloc or parsed.path)
    return scheme.casefold() in {"mailto", "tel", "urn", "doi", "file", "x-devonthink-item", "obsidian", "things", "bear"}


def _render_csv(rows: list[dict[str, str | int]]) -> str:
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
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
