"""CSV export for source canonical URL conflicts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

from graph.export._report_csv import field_value, flatten_values, get, metadata, normalized_key, render_csv, sort_key, source_id, write_csv

_FIELDNAMES = ["conflict_key", "source_count", "raw_urls", "source_ids"]
_URL_KEYS = {"canonical_url", "canonical", "url", "source_url", "external_url", "link", "permalink"}


def export_source_canonical_url_conflict_csv(
    sources: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write canonical URL groups that have conflicting source records."""
    source_list = list(sources)
    rows = _conflict_rows(source_list)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "source_count": len(source_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _conflict_rows(sources: list[Mapping[str, Any] | object]) -> list[dict[str, str | int]]:
    buckets: dict[str, dict[str, Any]] = defaultdict(lambda: {"source_ids": set(), "raw_urls": set()})
    for source in sources:
        candidates = _url_candidates(source)
        for raw_url in candidates:
            normalized = _normalize_url(raw_url)
            if not normalized:
                continue
            if source_id(source):
                buckets[normalized]["source_ids"].add(source_id(source))
            buckets[normalized]["raw_urls"].add(raw_url)

    rows: list[dict[str, str | int]] = []
    for conflict_key, bucket in buckets.items():
        if len(bucket["source_ids"]) < 2 and len(bucket["raw_urls"]) < 2:
            continue
        rows.append(
            {
                "conflict_key": conflict_key,
                "source_count": len(bucket["source_ids"]),
                "raw_urls": "; ".join(sorted(bucket["raw_urls"], key=sort_key)),
                "source_ids": "; ".join(sorted(bucket["source_ids"], key=sort_key)),
            }
        )
    return sorted(rows, key=lambda row: (-int(row["source_count"]), sort_key(row["conflict_key"])))


def _url_candidates(source: Mapping[str, Any] | object) -> list[str]:
    values: list[str] = []
    for key in _URL_KEYS:
        text = field_value(get(source, key))
        if text:
            values.append(text)
    for raw_key, value in metadata(source).items():
        if normalized_key(raw_key) in _URL_KEYS:
            values.extend(field_value(item) for item in flatten_values(value) if field_value(item))
    return values


def _normalize_url(value: str) -> str:
    parsed = urlparse(value.strip())
    if not parsed.scheme or not parsed.netloc:
        return ""
    scheme = parsed.scheme.casefold()
    hostname = (parsed.hostname or "").casefold()
    port = f":{parsed.port}" if parsed.port else ""
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((scheme, f"{hostname}{port}", path, "", "", ""))

