"""CSV export for inline Markdown code spans in unit content."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "inline_code_span_count", "max_inline_code_chars", "empty_inline_code_count", "has_shell_like_inline_code"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_SHELL_PREFIXES = ("$", ">", "pip", "python", "git", "npm", "pnpm", "uv", "pytest")


def export_units_to_inline_code_inventory_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = sorted((_row(unit) for unit in unit_list), key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(unit: Mapping[str, Any] | object) -> dict[str, str | int]:
    spans = _spans(_strip_fenced("" if get(unit, "content") is None else str(get(unit, "content"))))
    return {
        "unit_id": unit_id(unit),
        "inline_code_span_count": len(spans),
        "max_inline_code_chars": max((len(span) for span in spans), default=0),
        "empty_inline_code_count": sum(1 for span in spans if not span),
        "has_shell_like_inline_code": "true" if any(_shell_like(span) for span in spans) else "false",
    }


def _strip_fenced(content: str) -> str:
    kept: list[str] = []
    fence = ""
    for line in content.splitlines():
        match = _FENCE_RE.match(line)
        marker = match.group(1) if match else ""
        if marker and not fence:
            fence = marker[0]
            continue
        if fence and line.lstrip().startswith(fence * 3):
            fence = ""
            continue
        if not fence:
            kept.append(line)
    return "\n".join(kept)


def _spans(text: str) -> list[str]:
    spans: list[str] = []
    index = 0
    while index < len(text):
        if text[index] != "`":
            index += 1
            continue
        end = index
        while end < len(text) and text[end] == "`":
            end += 1
        ticks = end - index
        close = text.find("`" * ticks, end)
        if close == -1:
            if ticks >= 2:
                spans.append("")
            index = end
            continue
        spans.append(text[end:close])
        index = close + ticks
    return spans


def _shell_like(span: str) -> bool:
    text = field_value(span).casefold()
    return any(text == prefix or text.startswith(f"{prefix} ") for prefix in _SHELL_PREFIXES if prefix not in {"$", ">"}) or text.startswith(("$", ">"))
