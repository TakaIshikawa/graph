"""CSV export for Markdown-embedded HTML list structure elements."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, bool_attr, content_without_fences, line_number, preview, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "tag", "list_type", "start", "reversed", "item_count", "text_preview", "nesting_depth", "id", "class"]
_TAG_RE = re.compile(r"<(?P<closing>/)?(?P<tag>ul|ol|li)\b(?P<attrs>[^>]*)>", re.IGNORECASE)
_LI_RE = re.compile(r"<li\b", re.IGNORECASE)


def export_units_to_markdown_html_list_structure_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    content = content_without_fences(unit)
    context = unit_context(unit)
    rows: list[dict[str, str | int]] = []
    depth = 0
    for match in _TAG_RE.finditer(content):
        tag = match.group("tag").casefold()
        if match.group("closing"):
            if tag in {"ul", "ol"}:
                depth = max(0, depth - 1)
            continue
        values = attrs(match.group("attrs"))
        if tag in {"ul", "ol"}:
            body = _container_body(content, match.end(), tag)
            rows.append({**context, "line_number": line_number(content, match.start()), "tag": tag, "list_type": "ordered" if tag == "ol" else "unordered", "start": values.get("start", ""), "reversed": bool_attr(values, "reversed"), "item_count": len(_LI_RE.findall(body)), "text_preview": "", "nesting_depth": depth, "id": values.get("id", ""), "class": values.get("class", "")})
            depth += 1
        else:
            body = _li_body(content, match.end())
            rows.append({**context, "line_number": line_number(content, match.start()), "tag": tag, "list_type": "", "start": "", "reversed": "", "item_count": "", "text_preview": preview(body), "nesting_depth": depth, "id": values.get("id", ""), "class": values.get("class", "")})
    return rows


def _container_body(content: str, start: int, tag: str) -> str:
    match = re.search(rf"</{tag}\s*>", content[start:], re.IGNORECASE)
    return content[start : start + match.start()] if match else ""


def _li_body(content: str, start: int) -> str:
    match = re.search(r"</li\s*>", content[start:], re.IGNORECASE)
    return content[start : start + match.start()] if match else ""
