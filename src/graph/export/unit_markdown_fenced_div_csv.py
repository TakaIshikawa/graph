"""CSV export for Markdown fenced div blocks."""

from __future__ import annotations

import re
import shlex
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "source", "line_number", "opening_marker", "raw_info", "div_type", "class_names", "id_value", "closed"]
_DIV_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<marker>:{3,})(?P<info>[ \t].*)?$")
_CODE_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
_ATTR_TOKEN_RE = re.compile(r"[.#][A-Za-z0-9_-]+")


def export_units_to_markdown_fenced_div_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["raw_info"])))
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
    rows: list[dict[str, str | int | bool]] = []
    stack: list[dict[str, str | int | bool]] = []
    in_code_fence = False
    for line_number, line in enumerate(str(get(unit, "content") or data.get("content") or "").splitlines(), start=1):
        if _CODE_FENCE_RE.match(line):
            in_code_fence = not in_code_fence
            continue
        if in_code_fence:
            continue
        match = _DIV_RE.match(line)
        if not match or len(match.group("indent").replace("\t", "    ")) >= 4:
            continue
        info = field_value(match.group("info"))
        if not info:
            if stack:
                stack.pop()["closed"] = True
            continue
        row = _row(uid, title, source, line_number, match.group("marker"), info)
        rows.append(row)
        stack.append(row)
    return rows


def _row(uid: str, title: str, source: str, line_number: int, marker: str, info: str) -> dict[str, str | int | bool]:
    div_type, classes, id_value = _parse_info(info)
    return {
        "unit_id": uid,
        "title": title,
        "source": source,
        "line_number": line_number,
        "opening_marker": marker,
        "raw_info": info,
        "div_type": div_type,
        "class_names": " ".join(classes),
        "id_value": id_value,
        "closed": False,
    }


def _parse_info(info: str) -> tuple[str, list[str], str]:
    text = info.strip()
    try:
        tokens = shlex.split(text.replace("{", " { ").replace("}", " } "))
    except ValueError:
        tokens = text.split()
    div_type = ""
    classes: list[str] = []
    id_value = ""
    for token in tokens:
        if token in {"{", "}"}:
            continue
        for attr in _ATTR_TOKEN_RE.findall(token):
            if attr.startswith("."):
                classes.append(attr[1:])
            elif attr.startswith("#") and not id_value:
                id_value = attr[1:]
        bare = token.strip("{}")
        if bare and not bare.startswith((".", "#")) and "=" not in bare and not div_type:
            div_type = bare.casefold()
    return div_type, classes, id_value
