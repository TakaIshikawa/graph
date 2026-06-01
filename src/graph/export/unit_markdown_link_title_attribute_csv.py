"""CSV inventory for inline Markdown link title attributes."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "link_text", "target_url", "title_text", "line_number", "empty_title"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_LINK_RE = re.compile(r"(?<!!)\[([^\]\n]+)\]\((\S+)(?:\s+((?:\"[^\"]*\"|'[^']*'|\([^)]*\))))\)")


def export_unit_markdown_link_title_attributes_to_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write one row per inline Markdown link with a title attribute."""
    unit_list = list(units)
    rows: list[dict[str, str | int]] = []
    for unit in unit_list:
        title = field_value(get(unit, "title") or metadata(unit).get("title"))
        rows.extend({"unit_id": unit_id(unit), "title": title, **row} for row in _link_rows(_content(unit)))
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["target_url"]), sort_key(row["link_text"])))
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
        for match in _LINK_RE.finditer(line):
            title = _strip_title(match.group(3))
            rows.append(
                {
                    "link_text": field_value(match.group(1)),
                    "target_url": field_value(match.group(2)),
                    "title_text": field_value(title),
                    "line_number": line_number,
                    "empty_title": "true" if not title.strip() else "false",
                }
            )
    return rows


def _strip_title(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and ((text[0] == text[-1] and text[0] in {"'", '"'}) or (text[0] == "(" and text[-1] == ")")):
        return text[1:-1]
    return text
