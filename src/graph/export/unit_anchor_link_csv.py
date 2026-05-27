"""CSV export for Markdown links that target fragment anchors."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "link_text", "destination", "target_slug", "line_number", "matched_heading", "unresolved"]
_MD_LINK_RE = re.compile(r"!?\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_PUNCT_RE = re.compile(r"[^\w\s-]", re.UNICODE)


def export_units_to_anchor_link_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows: list[dict[str, str | int]] = []
    for unit in unit_list:
        rows.extend(_rows(unit))
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["target_slug"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    content = str(get(unit, "content") or "")
    headings = _headings(content)
    rows: list[dict[str, str | int]] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        for match in _MD_LINK_RE.finditer(line):
            destination = match.group(2)
            fragment = urlparse(destination).fragment
            if not fragment:
                continue
            slug = _slug(unquote(fragment))
            matched = headings.get(slug, "")
            rows.append(
                {
                    "unit_id": unit_id(unit),
                    "title": title,
                    "link_text": field_value(match.group(1)),
                    "destination": destination,
                    "target_slug": slug,
                    "line_number": line_number,
                    "matched_heading": matched,
                    "unresolved": "false" if matched else "true",
                }
            )
    return rows


def _headings(content: str) -> dict[str, str]:
    headings: dict[str, str] = {}
    in_fence = False
    for line in content.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _HEADING_RE.match(line)
        if match:
            text = field_value(match.group(2))
            headings.setdefault(_slug(text), text)
    return headings


def _slug(text: str) -> str:
    return re.sub(r"\s+", "-", _PUNCT_RE.sub("", text).casefold().strip()).strip("-")
