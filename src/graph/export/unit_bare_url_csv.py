"""CSV export for bare HTTP URLs in unit content."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "url", "scheme", "domain", "line_number", "surrounding_text"]
_URL_RE = re.compile(r"https?://[^\s<>\]\"']+", re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


def export_units_to_bare_url_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["url"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    for line_number, line in _content_lines(unit):
        for match in _URL_RE.finditer(line):
            url = match.group(0).rstrip(".,;:!?)]}")
            end = match.start() + len(url)
            if _is_markdown_url(line, match.start(), end):
                continue
            parsed = urlparse(url)
            scheme = parsed.scheme.casefold()
            domain = (parsed.hostname or "").casefold()
            if scheme not in {"http", "https"} or not domain:
                continue
            rows.append({"unit_id": uid, "title": title, "url": url, "scheme": scheme, "domain": domain, "line_number": line_number, "surrounding_text": field_value(line)})
    return rows


def _content_lines(unit: Mapping[str, Any] | object) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    in_fence = False
    for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            rows.append((line_number, line))
    return rows


def _is_markdown_url(line: str, start: int, end: int) -> bool:
    before = line[:start]
    after = line[end:]
    return (before.endswith("(") and after.startswith(")")) or (before.endswith("<") and after.startswith(">"))
