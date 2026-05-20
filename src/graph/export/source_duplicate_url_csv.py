"""CSV export for duplicate normalized URLs across sources."""

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

_FIELDNAMES = ["normalized_url", "domain", "unit_count", "sources", "unit_ids", "titles"]
_URL_KEYS = ("url", "urls", "link", "links", "href", "source_url", "canonical_url", "external_url", "web_url")
_URL_RE = re.compile(r"https?://[^\s<>)\"']+")
_WHITESPACE_RE = re.compile(r"\s+")


def export_source_duplicate_url_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write normalized URLs that appear in more than one unit."""
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
    groups: dict[str, dict[str, KnowledgeUnit | Mapping[str, Any]]] = defaultdict(dict)
    for unit in units:
        unit_id = _unit_id(unit)
        for raw_url in _unit_urls(unit):
            normalized_url = _normalize_url(raw_url)
            if normalized_url and unit_id:
                groups[normalized_url][unit_id] = unit

    rows: list[dict[str, str | int]] = []
    for normalized_url, units_by_id in groups.items():
        if len(units_by_id) <= 1:
            continue
        units_for_url = list(units_by_id.values())
        rows.append(
            {
                "normalized_url": normalized_url,
                "domain": urlsplit(normalized_url).hostname or "",
                "unit_count": len(units_by_id),
                "sources": _joined_unique(_field_value(_get(unit, "source_project")) or "Unknown" for unit in units_for_url),
                "unit_ids": _joined_unique(units_by_id),
                "titles": _joined_unique(_field_value(_get(unit, "title")) for unit in units_for_url),
            }
        )
    return sorted(rows, key=lambda row: (-int(row["unit_count"]), _sort_key(row["normalized_url"])))


def _unit_urls(unit: KnowledgeUnit | Mapping[str, Any]) -> list[str]:
    urls: list[str] = []
    for key in _URL_KEYS:
        urls.extend(_string_values(_get(unit, key)))
    urls.extend(_metadata_urls(_metadata(unit)))
    urls.extend(_URL_RE.findall(_field_value(_get(unit, "content"))))
    return sorted(set(urls), key=_sort_key)


def _metadata_urls(metadata: Mapping[str, Any]) -> list[str]:
    urls: list[str] = []
    for key, value in metadata.items():
        if _normalized_key(key) in _URL_KEYS:
            urls.extend(_string_values(value))
        elif isinstance(value, Mapping):
            urls.extend(_metadata_urls(value))
        elif isinstance(value, list | tuple | set):
            for item in value:
                if isinstance(item, Mapping):
                    urls.extend(_metadata_urls(item))
    return urls


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
    text = raw_url.strip().rstrip(".,;:!?)]}")
    parsed = urlsplit(text)
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
        return ""
    path = parsed.path
    if path != "/":
        path = path.rstrip("/")
    else:
        path = ""
    return urlunsplit(SplitResult(parsed.scheme.lower(), parsed.netloc.lower(), path, parsed.query, ""))


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _unit_id(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "id")) or _field_value(_get(unit, "source_id"))


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _joined_unique(values: Iterable[object]) -> str:
    return "; ".join(sorted({_field_value(value) for value in values if _field_value(value)}, key=_sort_key))


def _normalized_key(value: object) -> str:
    return _inline_text(value).casefold().replace("-", "_").replace(" ", "_")


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
