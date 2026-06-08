"""CSV export for Markdown-embedded HTML iframe sandbox security metadata."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, content_without_fences, line_number, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "src", "title", "sandbox", "sandbox_token_count", "allows_scripts", "allows_same_origin", "allow", "referrerpolicy", "loading", "missing_title", "id", "class"]
_IFRAME_RE = re.compile(r"<iframe\b(?P<attrs>[^>]*)>", re.IGNORECASE)


def export_units_to_markdown_html_iframe_sandbox_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
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
    for match in _IFRAME_RE.finditer(content):
        values = attrs(match.group("attrs"))
        sandbox_tokens = {token.casefold() for token in values.get("sandbox", "").split()}
        rows.append({**context, "line_number": line_number(content, match.start()), "src": values.get("src", ""), "title": values.get("title", ""), "sandbox": values.get("sandbox", ""), "sandbox_token_count": len(sandbox_tokens), "allows_scripts": str("allow-scripts" in sandbox_tokens).lower(), "allows_same_origin": str("allow-same-origin" in sandbox_tokens).lower(), "allow": values.get("allow", ""), "referrerpolicy": values.get("referrerpolicy", ""), "loading": values.get("loading", ""), "missing_title": str(not values.get("title", "")).lower(), "id": values.get("id", ""), "class": values.get("class", "")})
    return rows
