"""CSV export for source URL duplicates across units."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import SplitResult, urlsplit, urlunsplit

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "normalized_url",
    "unit_count",
    "source_project_count",
    "source_projects",
    "source_entity_types",
    "unit_ids",
    "titles",
    "raw_urls",
]
_URL_KEYS = ("url", "link", "href", "source_url", "canonical_url", "external_url")
_WHITESPACE_RE = re.compile(r"\s+")


def export_source_url_duplicates_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write rows for URL values reused by multiple distinct units."""
    unit_list = list(units)
    rows = _duplicate_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "duplicate_url_count": len(rows),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _duplicate_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    groups: dict[str, list[tuple[KnowledgeUnit | Mapping[str, Any], str]]] = defaultdict(list)
    for unit in units:
        for raw_url in _unit_urls(unit):
            normalized_url = _normalize_url(raw_url)
            if normalized_url:
                groups[normalized_url].append((unit, raw_url))

    rows: list[dict[str, str | int]] = []
    for normalized_url, entries in groups.items():
        unit_ids = {_field_value(_get(unit, "id")) for unit, _ in entries if _field_value(_get(unit, "id"))}
        if len(unit_ids) <= 1:
            continue

        units_by_id = {unit_id: unit for unit, _ in entries if (unit_id := _field_value(_get(unit, "id")))}
        rows.append(
            {
                "normalized_url": normalized_url,
                "unit_count": len(unit_ids),
                "source_project_count": len(
                    {_field_value(_get(unit, "source_project")) or "Unknown" for unit in units_by_id.values()}
                ),
                "source_projects": _joined_unique(
                    _field_value(_get(unit, "source_project")) or "Unknown" for unit in units_by_id.values()
                ),
                "source_entity_types": _joined_unique(
                    _field_value(_get(unit, "source_entity_type")) or "Unknown" for unit in units_by_id.values()
                ),
                "unit_ids": _joined_unique(unit_ids),
                "titles": _joined_unique(_field_value(_get(unit, "title")) for unit in units_by_id.values()),
                "raw_urls": _joined_unique(raw_url for _, raw_url in entries),
            }
        )

    return sorted(rows, key=lambda row: (-int(row["unit_count"]), _sort_key(row["normalized_url"])))


def _unit_urls(unit: KnowledgeUnit | Mapping[str, Any]) -> list[str]:
    urls: list[str] = []
    for key in _URL_KEYS:
        urls.extend(_string_values(_get(unit, key)))
    metadata = _metadata(unit)
    for key in _URL_KEYS:
        urls.extend(_string_values(metadata.get(key)))
    return _unique_sorted(urls)


def _string_values(value: object) -> list[str]:
    if value is None or isinstance(value, bytes):
        return []
    if isinstance(value, str):
        text = _inline_text(value)
        return [text] if text else []
    if isinstance(value, Mapping):
        return []
    if isinstance(value, Iterable):
        values: list[str] = []
        for item in value:
            values.extend(_string_values(item))
        return values
    return []


def _normalize_url(raw_url: str) -> str:
    parsed = urlsplit(raw_url.strip())
    if not parsed.scheme or not parsed.netloc:
        return ""
    path = parsed.path
    if path != "/":
        path = path.rstrip("/")
    else:
        path = ""
    return urlunsplit(
        SplitResult(
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path,
            parsed.query,
            "",
        )
    )


def _metadata(value: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(value, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _joined_unique(values: Iterable[object]) -> str:
    return "; ".join(_unique_sorted(_field_value(value) for value in values))


def _unique_sorted(values: Iterable[str]) -> list[str]:
    return sorted({value for value in values if value}, key=_sort_key)


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
