"""CSV export for Markdown image title attributes."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "source", "line_number", "alt_text", "image_url", "title_attribute", "quote_style"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_IMAGE_START_RE = re.compile(r"!\[")


def export_units_to_markdown_image_title_attribute_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write one row per Markdown image with an explicit title attribute."""
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["image_url"]), sort_key(row["alt_text"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    meta = metadata(unit)
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or meta.get("title"))
    source = field_value(get(unit, "source") or get(unit, "source_project") or meta.get("source") or meta.get("source_project"))
    rows: list[dict[str, str | int]] = []
    in_fence = False

    for line_number, line in enumerate(_content(unit).splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for image in _image_title_attributes(line):
            rows.append({"unit_id": uid, "title": title, "source": source, "line_number": line_number, **image})
    return rows


def _content(unit: Mapping[str, Any] | object) -> str:
    return str(get(unit, "content") or metadata(unit).get("content") or "")


def _image_title_attributes(line: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    code_spans = _inline_code_spans(line)
    for match in _IMAGE_START_RE.finditer(line):
        if _inside_spans(match.start(), code_spans):
            continue
        parsed = _parse_image(line, match.start())
        if parsed is not None:
            rows.append(parsed)
    return rows


def _parse_image(line: str, start: int) -> dict[str, str] | None:
    alt_end = _find_closing(line, start + 2, "]")
    if alt_end is None or alt_end + 1 >= len(line) or line[alt_end + 1] != "(":
        return None
    target_end = _find_image_destination_end(line, alt_end + 2)
    if target_end is None:
        return None
    raw_target = line[alt_end + 2 : target_end].strip()
    cursor = _skip_spaces(line, target_end)
    if cursor >= len(line) or line[cursor] == ")":
        return None
    title = _parse_title(line, cursor)
    if title is None:
        return None
    title_text, quote_style, cursor = title
    cursor = _skip_spaces(line, cursor)
    if cursor >= len(line) or line[cursor] != ")":
        return None
    return {
        "alt_text": field_value(_unescape(line[start + 2 : alt_end])),
        "image_url": field_value(_unescape(raw_target)),
        "title_attribute": field_value(_unescape(title_text)),
        "quote_style": quote_style,
    }


def _find_closing(line: str, start: int, closer: str) -> int | None:
    escaped = False
    for index in range(start, len(line)):
        char = line[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == closer:
            return index
    return None


def _find_image_destination_end(line: str, start: int) -> int | None:
    escaped = False
    for index in range(start, len(line)):
        char = line[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char.isspace() or char == ")":
            return index
    return None


def _parse_title(line: str, start: int) -> tuple[str, str, int] | None:
    opener = line[start]
    if opener not in {"'", '"', "("}:
        return None
    closer = ")" if opener == "(" else opener
    quote_style = {"'": "single", '"': "double", "(": "parentheses"}[opener]
    escaped = False
    for index in range(start + 1, len(line)):
        char = line[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == closer:
            return line[start + 1 : index], quote_style, index + 1
    return None


def _skip_spaces(line: str, start: int) -> int:
    while start < len(line) and line[start].isspace():
        start += 1
    return start


def _inline_code_spans(line: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(line):
        if line[index] != "`":
            index += 1
            continue
        tick_count = 1
        while index + tick_count < len(line) and line[index + tick_count] == "`":
            tick_count += 1
        closer = line.find("`" * tick_count, index + tick_count)
        if closer == -1:
            break
        spans.append((index, closer + tick_count))
        index = closer + tick_count
    return spans


def _inside_spans(index: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= index < end for start, end in spans)


def _unescape(value: str) -> str:
    return re.sub(r"\\(.)", r"\1", value)
