"""CSV export for Markdown emphasis marker counts by unit."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import get, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "bold_count", "italic_count", "strikethrough_count", "highlight_count"]
_BOLD_RE = re.compile(r"(?<!\\)(?:\*\*|__)(?=\S)(.+?)(?<=\S)(?:\*\*|__)", re.DOTALL)
_ITALIC_RE = re.compile(r"(?<![\*_\\])(?:\*|_)(?=\S)(.+?)(?<=\S)(?:\*|_)(?![\*_])", re.DOTALL)
_STRIKE_RE = re.compile(r"(?<!\\)~~(?=\S)(.+?)(?<=\S)~~", re.DOTALL)
_HIGHLIGHT_RE = re.compile(r"(?<!\\)==(?=\S)(.+?)(?<=\S)==", re.DOTALL)


def export_units_to_markdown_emphasis_inventory_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [_row(unit) for unit in unit_list]
    rows.sort(key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(unit: Mapping[str, Any] | object) -> dict[str, str | int]:
    text = _strip_code(str(get(unit, "content") or ""))
    return {"unit_id": unit_id(unit), "bold_count": len(_BOLD_RE.findall(text)), "italic_count": len(_ITALIC_RE.findall(_BOLD_RE.sub("", text))), "strikethrough_count": len(_STRIKE_RE.findall(text)), "highlight_count": len(_HIGHLIGHT_RE.findall(text))}


def _strip_code(content: str) -> str:
    lines: list[str] = []
    in_fence = False
    for line in content.splitlines():
        if line.lstrip().startswith("```") or line.lstrip().startswith("~~~"):
            in_fence = not in_fence
            continue
        if not in_fence:
            lines.append(line)
    return re.sub(r"`[^`\n]*`", "", "\n".join(lines))
