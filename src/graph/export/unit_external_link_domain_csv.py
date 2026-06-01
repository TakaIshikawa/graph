"""CSV inventory for external Markdown link domains."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "url", "domain", "registrable_host", "line_number", "link_text", "domain_count_within_unit"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_LINK_RE = re.compile(r"(?<!!)\[([^\]\n]+)\]\((\S+?)(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?\)")
_AUTOLINK_RE = re.compile(r"<(https?://[^<>\s]+)>", re.IGNORECASE)
_SECOND_LEVEL_SUFFIXES = {"co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "net.au", "org.au", "co.jp", "com.br"}


def export_unit_external_link_domains_to_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write one row per external http/https Markdown or autolink URL."""
    unit_list = list(units)
    rows: list[dict[str, str | int]] = []
    for unit in unit_list:
        title = field_value(get(unit, "title") or metadata(unit).get("title"))
        unit_rows = _link_rows(_content(unit))
        counts = Counter(row["domain"] for row in unit_rows)
        rows.extend({"unit_id": unit_id(unit), "title": title, **row, "domain_count_within_unit": counts[row["domain"]]} for row in unit_rows)
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["domain"]), sort_key(row["url"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _content(unit: Mapping[str, Any] | object) -> str:
    return str(get(unit, "content") or metadata(unit).get("content") or "")


def _link_rows(content: str) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for text, url in [(m.group(1), m.group(2)) for m in _LINK_RE.finditer(line)] + [(m.group(1), m.group(1)) for m in _AUTOLINK_RE.finditer(line)]:
            parsed = urlparse(url)
            if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
                continue
            domain = (parsed.hostname or "").casefold()
            rows.append({"url": field_value(url), "domain": domain, "registrable_host": _registrable_host(domain), "line_number": line_number, "link_text": field_value(text)})
    return rows


def _registrable_host(domain: str) -> str:
    parts = [part for part in domain.split(".") if part]
    if len(parts) <= 2:
        return domain
    suffix = ".".join(parts[-2:])
    if suffix in _SECOND_LEVEL_SUFFIXES and len(parts) >= 3:
        return ".".join(parts[-3:])
    return suffix
