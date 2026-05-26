"""CSV export for fenced code blocks in unit Markdown content."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "code_block_count", "languages", "unlabeled_block_count", "max_block_line_count"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})\s*([^`]*)$")


def export_units_to_code_block_inventory_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = sorted((_row(unit) for unit in unit_list), key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(unit: Mapping[str, Any] | object) -> dict[str, str | int]:
    blocks = _blocks("" if get(unit, "content") is None else str(get(unit, "content")))
    languages = sorted({language for language, _lines in blocks if language}, key=sort_key)
    return {
        "unit_id": unit_id(unit),
        "code_block_count": len(blocks),
        "languages": "; ".join(languages),
        "unlabeled_block_count": sum(1 for language, _lines in blocks if not language),
        "max_block_line_count": max((lines for _language, lines in blocks), default=0),
    }


def _blocks(content: str) -> list[tuple[str, int]]:
    blocks: list[tuple[str, int]] = []
    fence = ""
    language = ""
    line_count = 0
    for line in content.splitlines():
        match = _FENCE_RE.match(line)
        if match and not fence:
            fence = match.group(1)[0]
            info = field_value(match.group(2))
            language = field_value(info.split(maxsplit=1)[0] if info else "").casefold()
            line_count = 0
            continue
        if fence and line.lstrip().startswith(fence * 3):
            blocks.append((language, line_count))
            fence = ""
            continue
        if fence:
            line_count += 1
    if fence:
        blocks.append((language, line_count))
    return blocks
