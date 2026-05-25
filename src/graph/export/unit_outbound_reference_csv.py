"""CSV export for outbound references in unit content and metadata."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.parse import urlunparse

from graph.export._report_csv import field_value, flatten_values, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "urls", "domains", "markdown_links", "wikilinks", "citations", "reference_count"]
_URL_RE = re.compile(r"\bhttps?://[^\s<>'\")\]]+")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
_CITATION_RE = re.compile(r"(?<!\w)@([A-Za-z][A-Za-z0-9_:-]+)")


def export_units_to_outbound_reference_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = sorted((_row(unit) for unit in unit_list), key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(unit: Mapping[str, Any] | object) -> dict[str, str | int]:
    text = "\n".join(field_value(value) for value in [get(unit, "content"), *flatten_values(metadata(unit))])
    urls = {_clean(match) for match in _URL_RE.findall(text)}
    markdown_links = {f"{label}|{_clean(url)}" for label, url in _MD_LINK_RE.findall(text)}
    urls.update(link.split("|", 1)[1] for link in markdown_links)
    wikilinks = {field_value(value) for value in _WIKILINK_RE.findall(text)}
    citations = {f"@{value}" for value in _CITATION_RE.findall(text)}
    domains = {urlparse(url).netloc.casefold() for url in urls if urlparse(url).netloc}
    references = urls | wikilinks | citations
    return {
        "unit_id": unit_id(unit),
        "urls": "; ".join(sorted(urls, key=sort_key)),
        "domains": "; ".join(sorted(domains, key=sort_key)),
        "markdown_links": "; ".join(sorted(markdown_links, key=sort_key)),
        "wikilinks": "; ".join(sorted(wikilinks, key=sort_key)),
        "citations": "; ".join(sorted(citations, key=sort_key)),
        "reference_count": len(references),
    }


def _clean(value: str) -> str:
    text = field_value(value).rstrip(".,;:")
    parsed = urlparse(text)
    if parsed.scheme and parsed.netloc:
        return urlunparse((parsed.scheme.casefold(), parsed.netloc.casefold(), parsed.path, "", parsed.query, parsed.fragment))
    return text
