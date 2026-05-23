"""CSV export for source URL hostname inventory."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import field_value, flatten_values, get, metadata, normalized_key, render_csv, sort_key, source_id, write_csv

_FIELDNAMES = ["hostname", "source_count", "source_ids", "sample_urls"]
_URL_KEYS = {"url", "source_url", "canonical_url", "external_url", "homepage", "link", "permalink"}
_UNKNOWN = "unknown"
_SAMPLE_LIMIT = 3


def export_source_hostname_inventory_csv(
    sources: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write source counts grouped by parsed URL hostname."""
    source_list = list(sources)
    rows = _inventory_rows(source_list)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "source_count": len(source_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _inventory_rows(sources: list[Mapping[str, Any] | object]) -> list[dict[str, str | int]]:
    buckets: dict[str, dict[str, Any]] = defaultdict(lambda: {"source_ids": set(), "urls": []})
    for source in sources:
        hostnames = []
        for url in _source_urls(source):
            hostname = _hostname(url)
            hostnames.append(hostname)
            if url:
                buckets[hostname]["urls"].append(url)
        if not hostnames:
            hostnames = [_UNKNOWN]
        for hostname in set(hostnames):
            if source_id(source):
                buckets[hostname]["source_ids"].add(source_id(source))

    rows: list[dict[str, str | int]] = []
    for hostname, bucket in buckets.items():
        rows.append(
            {
                "hostname": hostname,
                "source_count": len(bucket["source_ids"]),
                "source_ids": "; ".join(sorted(bucket["source_ids"], key=sort_key)),
                "sample_urls": "; ".join(bucket["urls"][:_SAMPLE_LIMIT]),
            }
        )
    return sorted(rows, key=lambda row: (-int(row["source_count"]), sort_key(row["hostname"])))


def _source_urls(source: Mapping[str, Any] | object) -> list[str]:
    urls: list[str] = []
    for key in _URL_KEYS:
        text = field_value(get(source, key))
        if text:
            urls.append(text)
    for raw_key, value in metadata(source).items():
        if normalized_key(raw_key) in _URL_KEYS:
            urls.extend(field_value(item) for item in flatten_values(value) if field_value(item))
    return urls


def _hostname(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme and "." in parsed.path:
        parsed = urlparse(f"//{url}")
    return parsed.hostname.casefold() if parsed.hostname else _UNKNOWN
