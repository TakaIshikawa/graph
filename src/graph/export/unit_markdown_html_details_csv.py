"""CSV export for HTML details blocks embedded in unit Markdown content."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "source", "line_number", "is_open", "summary_text"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_OPEN_RE = re.compile(r"<details\b(?P<attrs>[^>]*)>", re.IGNORECASE)
_CLOSE_RE = re.compile(r"</details\s*>", re.IGNORECASE)
_SUMMARY_RE = re.compile(r"<summary\b[^>]*>(?P<text>.*?)</summary\s*>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")


def export_unit_markdown_html_details_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    """Return or write one row per HTML details block."""
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int | bool]]:
    data = metadata(unit)
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or data.get("title"))
    source = field_value(get(unit, "source") or get(unit, "source_url") or data.get("source") or data.get("source_url"))
    rows: list[dict[str, str | int | bool]] = []
    in_fence = False
    active: dict[str, str | int | bool] | None = None
    block: list[str] = []
    for line_number, line in enumerate(str(get(unit, "content") or data.get("content") or "").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if active is None:
            match = _OPEN_RE.search(line)
            if not match:
                continue
            active = {
                "unit_id": uid,
                "title": title,
                "source": source,
                "line_number": line_number,
                "is_open": _has_open_attribute(match.group("attrs")),
            }
            block = [line[match.start() :]]
        else:
            block.append(line)
        if active is not None and _CLOSE_RE.search(line):
            rows.append({**active, "summary_text": _summary_text("\n".join(block))})
            active = None
            block = []
    if active is not None:
        rows.append({**active, "summary_text": _summary_text("\n".join(block))})
    return rows


def _has_open_attribute(attrs: str) -> bool:
    return bool(re.search(r"""(?:^|\s)open(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s"'=<>`]+))?(?=\s|$)""", attrs, re.IGNORECASE))


def _summary_text(block: str) -> str:
    match = _SUMMARY_RE.search(block)
    if not match:
        return ""
    return field_value(_TAG_RE.sub(" ", match.group("text")))
