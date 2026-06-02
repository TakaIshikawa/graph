"""CSV export for per-unit outbound domain inventory."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import field_value, flatten_values, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "domain", "url_count", "sample_url"]
_URL_RE = re.compile(r"\bhttps?://[^\s<>'\"]+")


def export_units_to_outbound_domain_inventory_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), sort_key(row["domain"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for url in _urls(unit):
        host = urlparse(url).hostname
        if host:
            groups[host.casefold()].append(url)
    return [
        {"unit_id": unit_id(unit), "title": field_value(get(unit, "title") or metadata(unit).get("title")), "domain": domain, "url_count": len(urls), "sample_url": urls[0]}
        for domain, urls in groups.items()
    ]


def _urls(unit: Mapping[str, Any] | object) -> list[str]:
    values = [get(unit, "content"), metadata(unit)]
    urls: list[str] = []
    for value in values:
        for item in flatten_values(value):
            for candidate in _URL_RE.findall(field_value(item)):
                url = candidate.rstrip(".,);]")
                if urlparse(url).scheme in {"http", "https"} and urlparse(url).hostname:
                    urls.append(url)
    return urls
