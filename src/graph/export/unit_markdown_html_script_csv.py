"""CSV export for Markdown-embedded HTML script elements."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, bool_attr, content_without_fences, domain, line_number, preview, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "src", "domain", "type", "async", "defer", "module", "nomodule", "integrity", "crossorigin", "referrerpolicy", "inline", "inline_preview"]
_SCRIPT_RE = re.compile(r"<script\b(?P<attrs>[^>]*)>(?P<body>.*?)</script\s*>|<script\b(?P<single_attrs>[^>]*)/?>", re.IGNORECASE | re.DOTALL)


def export_units_to_markdown_html_script_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["src"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    content = content_without_fences(unit)
    context = unit_context(unit)
    rows: list[dict[str, str | int]] = []
    for match in _SCRIPT_RE.finditer(content):
        body = match.group("body") or ""
        values = attrs(match.group("attrs") or match.group("single_attrs") or "")
        src = values.get("src", "")
        script_type = values.get("type", "")
        is_module = script_type.casefold() == "module"
        inline = not src and bool(preview(body))
        rows.append({**context, "line_number": line_number(content, match.start()), "src": src, "domain": domain(src), "type": script_type, "async": bool_attr(values, "async"), "defer": bool_attr(values, "defer"), "module": str(is_module).lower(), "nomodule": bool_attr(values, "nomodule"), "integrity": values.get("integrity", ""), "crossorigin": values.get("crossorigin", ""), "referrerpolicy": values.get("referrerpolicy", ""), "inline": str(inline).lower(), "inline_preview": preview(body) if inline else ""})
    return rows
