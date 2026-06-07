"""CSV export for Markdown-embedded HTML menu command elements."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, bool_attr, content_without_fences, line_number, preview, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "tag", "type", "label", "command", "icon", "checked", "disabled", "radiogroup", "text_preview"]
_MENU_RE = re.compile(r"<menu\b[^>]*>(?P<body>.*?)</menu\s*>", re.IGNORECASE | re.DOTALL)
_COMMAND_RE = re.compile(r"<(?P<tag>command|menuitem)\b(?P<attrs>[^>]*)/?>|<button\b(?P<button_attrs>[^>]*)>(?P<button_body>.*?)</button\s*>", re.IGNORECASE | re.DOTALL)


def export_units_to_markdown_html_menu_command_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["label"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    content = content_without_fences(unit)
    context = unit_context(unit)
    rows: list[dict[str, str | int]] = []
    for menu in _MENU_RE.finditer(content):
        body = menu.group("body")
        for command in _COMMAND_RE.finditer(body):
            tag = (command.group("tag") or "button").casefold()
            values = attrs(command.group("attrs") or command.group("button_attrs") or "")
            rows.append({**context, "line_number": line_number(content, menu.start()) + body.count("\n", 0, command.start()), "tag": tag, "type": values.get("type", ""), "label": values.get("label", "") or preview(command.group("button_body") or ""), "command": values.get("command", ""), "icon": values.get("icon", ""), "checked": bool_attr(values, "checked"), "disabled": bool_attr(values, "disabled"), "radiogroup": values.get("radiogroup", ""), "text_preview": preview(command.group("button_body") or "")})
    return rows
