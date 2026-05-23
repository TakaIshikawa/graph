"""CSV export for outbound HTTP(S) links referenced by units."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import flatten_values, get, metadata, render_csv, sort_key, unit_id, write_csv
from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["scheme", "hostname", "unit_count", "link_count", "unit_ids", "sample_urls"]
_URL_RE = re.compile(r"\bhttps?://[^\s<>'\"]+")
_SAMPLE_LIMIT = 3


def export_unit_outbound_link_inventory_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write outbound HTTP(S) links grouped by scheme and hostname."""
    unit_list = list(units)
    rows = _inventory_rows(unit_list)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _inventory_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    buckets: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"links": 0, "unit_ids": set(), "urls": []})
    for unit in units:
        seen_for_unit: set[tuple[str, str]] = set()
        for url in _unit_urls(unit):
            parsed = urlparse(url)
            if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
                continue
            key = (parsed.scheme.casefold(), parsed.hostname.casefold())
            buckets[key]["links"] += 1
            buckets[key]["urls"].append(url)
            seen_for_unit.add(key)
        for key in seen_for_unit:
            if unit_id(unit):
                buckets[key]["unit_ids"].add(unit_id(unit))

    rows: list[dict[str, str | int]] = []
    for scheme, hostname in sorted(buckets, key=lambda key: (sort_key(key[0]), sort_key(key[1]))):
        bucket = buckets[(scheme, hostname)]
        rows.append(
            {
                "scheme": scheme,
                "hostname": hostname,
                "unit_count": len(bucket["unit_ids"]),
                "link_count": bucket["links"],
                "unit_ids": "; ".join(sorted(bucket["unit_ids"], key=sort_key)),
                "sample_urls": "; ".join(bucket["urls"][:_SAMPLE_LIMIT]),
            }
        )
    return rows


def _unit_urls(unit: KnowledgeUnit | Mapping[str, Any]) -> list[str]:
    values = [get(unit, "content"), get(unit, "text"), get(unit, "html")]
    values.extend(flatten_values(metadata(unit)))
    urls: list[str] = []
    for value in values:
        urls.extend(candidate.rstrip(".,);]") for candidate in _URL_RE.findall(str(value)))
    return urls

