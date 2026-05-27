"""CSV export for external HTTP links in unit content and metadata."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import field_value, flatten_values, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "url", "domain", "source_field", "occurrence_count"]
_URL_RE = re.compile(r"https?://[^\s<>()\[\]\"']+", re.IGNORECASE)


def export_units_to_external_link_inventory_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows: list[dict[str, str | int]] = []
    for unit in unit_list:
        title = field_value(get(unit, "title") or metadata(unit).get("title"))
        counts: Counter[tuple[str, str]] = Counter()
        for url in _content_urls(field_value(get(unit, "content"))):
            counts[(url, "content")] += 1
        for source_field, url in _metadata_urls(metadata(unit)):
            counts[(url, source_field)] += 1
        for (url, source_field), count in counts.items():
            rows.append(
                {
                    "unit_id": unit_id(unit),
                    "title": title,
                    "url": url,
                    "domain": urlparse(url).netloc.casefold(),
                    "source_field": source_field,
                    "occurrence_count": count,
                }
            )
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), sort_key(row["url"]), sort_key(row["source_field"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _content_urls(content: str) -> list[str]:
    return [_clean_url(match.group(0)) for match in _URL_RE.finditer(content) if _is_external(_clean_url(match.group(0)))]


def _metadata_urls(meta: Mapping[str, Any], prefix: str = "metadata") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for key, value in meta.items():
        field = f"{prefix}.{field_value(key)}"
        if isinstance(value, Mapping):
            found.extend(_metadata_urls(value, field))
            continue
        for item in flatten_values(value):
            text = field_value(item)
            if _is_external(text):
                found.append((field, text))
    return found


def _clean_url(url: str) -> str:
    return url.rstrip(".,;:!?)\"]}'")


def _is_external(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme.casefold() in {"http", "https"} and bool(parsed.netloc)
