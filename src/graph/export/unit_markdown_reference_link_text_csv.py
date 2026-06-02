"""CSV export for Markdown reference-style link text usages."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "source", "line_number", "visible_text", "reference_label", "normalized_label", "usage_style", "definition_exists"]
_DEF_RE = re.compile(r"^[ \t]{0,3}\[(?P<label>[^\]\n]+)]:")
_FULL_RE = re.compile(r"(?<!!)\[(?P<text>[^\]\n]+)]\[(?P<label>[^\]\n]*)]")
_SHORTCUT_RE = re.compile(r"(?<!!)(?<!])\[(?P<label>[^\]\n]+)](?![\[(])")
_INLINE_RE = re.compile(r"(?<!!)\[[^\]\n]+]\([^)]*\)")
_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
_ORDER = {"full": 0, "collapsed": 1, "shortcut": 2}


def export_units_to_markdown_reference_link_text_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), _ORDER[str(row["usage_style"])], sort_key(row["reference_label"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int | bool]]:
    data = metadata(unit)
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or data.get("title"))
    source = field_value(get(unit, "source") or get(unit, "source_url") or data.get("source") or data.get("source_url"))
    content = str(get(unit, "content") or data.get("content") or "")
    definitions = {_normalize(match.group("label")) for line in content.splitlines() if (match := _DEF_RE.match(line))}
    rows: list[dict[str, str | int | bool]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence or _DEF_RE.match(line):
            continue
        masked = list(line)
        for inline in _INLINE_RE.finditer(line):
            masked[inline.start() : inline.end()] = " " * (inline.end() - inline.start())
        for match in _FULL_RE.finditer("".join(masked)):
            if _inside_code_span(line, match.start()):
                continue
            visible = field_value(match.group("text"))
            label = field_value(match.group("label")) or visible
            style = "collapsed" if not field_value(match.group("label")) else "full"
            rows.append(_row(uid, title, source, line_number, visible, label, style, definitions))
            masked[match.start() : match.end()] = " " * (match.end() - match.start())
        for match in _SHORTCUT_RE.finditer("".join(masked)):
            if _inside_code_span(line, match.start()):
                continue
            label = field_value(match.group("label"))
            rows.append(_row(uid, title, source, line_number, label, label, "shortcut", definitions))
    return rows


def _row(uid: str, title: str, source: str, line_number: int, visible: str, label: str, style: str, definitions: set[str]) -> dict[str, str | int | bool]:
    normalized = _normalize(label)
    return {"unit_id": uid, "title": title, "source": source, "line_number": line_number, "visible_text": visible, "reference_label": label, "normalized_label": normalized, "usage_style": style, "definition_exists": normalized in definitions}


def _normalize(label: str) -> str:
    return re.sub(r"\s+", " ", field_value(label)).casefold()


def _inside_code_span(line: str, offset: int) -> bool:
    return line[:offset].count("`") % 2 == 1
