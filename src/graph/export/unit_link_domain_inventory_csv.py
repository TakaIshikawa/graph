"""CSV export for per-unit link domain inventory."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import field_value, flatten_values, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "domain_count", "link_count", "top_domain", "external_domain_count"]
_URL_RE = re.compile(r"\bhttps?://[^\s<>'\"]+")


def export_units_to_link_domain_inventory_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [_row(unit) for unit in unit_list]
    rows.sort(key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(unit: Mapping[str, Any] | object) -> dict[str, str | int]:
    domains = [_domain(url) for url in _urls(unit)]
    counts = Counter(domain for domain in domains if domain)
    top_domain = sorted(counts.items(), key=lambda item: (-item[1], sort_key(item[0])))[0][0] if counts else ""
    return {
        "unit_id": unit_id(unit),
        "title": field_value(get(unit, "title") or metadata(unit).get("title")),
        "domain_count": len(counts),
        "link_count": sum(counts.values()),
        "top_domain": top_domain,
        "external_domain_count": len(counts),
    }


def _urls(unit: Mapping[str, Any] | object) -> list[str]:
    urls: list[str] = []
    for value in (get(unit, "content"), metadata(unit)):
        for item in flatten_values(value):
            for candidate in _URL_RE.findall(field_value(item)):
                url = candidate.rstrip(".,);]")
                if urlparse(url).scheme in {"http", "https"} and urlparse(url).hostname:
                    urls.append(url)
    return urls


def _domain(url: str) -> str:
    host = urlparse(url).hostname or ""
    host = host.casefold()
    return host[4:] if host.startswith("www.") else host
