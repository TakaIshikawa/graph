"""CSV export for reference-like YAML frontmatter fields."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "field_path", "reference_type", "value"]
_DOI_RE = re.compile(r"\b10\.\d{4,9}/\S+\b", re.IGNORECASE)
_ISBN_RE = re.compile(r"\b(?:97[89][-\s]?)?(?:\d[-\s]?){9}[\dX]\b", re.IGNORECASE)
_CITEKEY_RE = re.compile(r"^@[A-Za-z0-9_:.-]+$")
_WIKILINK_RE = re.compile(r"\[\[[^\]]+\]\]")
_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def export_units_to_frontmatter_reference_field_csv(
    units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), sort_key(row["field_path"]), sort_key(row["reference_type"]), sort_key(row["value"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str]]:
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    data = _frontmatter(str(get(unit, "content") or ""))
    rows: list[dict[str, str]] = []
    for path, value in _flatten(data):
        ref_type = _reference_type(value)
        if ref_type:
            rows.append({"unit_id": uid, "title": title, "field_path": path, "reference_type": ref_type, "value": field_value(str(value))})
    return rows


def _frontmatter(content: str) -> Any:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            try:
                return yaml.safe_load("\n".join(lines[1:index])) or {}
            except yaml.YAMLError:
                return {}
    return {}


def _flatten(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, Mapping):
        rows: list[tuple[str, Any]] = []
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten(child, path))
        return rows
    if isinstance(value, list):
        rows = []
        for index, child in enumerate(value):
            rows.extend(_flatten(child, f"{prefix}[{index}]"))
        return rows
    return [(prefix, value)]


def _reference_type(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""
    if _URL_RE.match(text):
        return "url"
    if _DOI_RE.search(text):
        return "doi"
    if _ISBN_RE.search(text):
        return "isbn"
    if _CITEKEY_RE.match(text):
        return "citekey"
    if _WIKILINK_RE.search(text):
        return "wikilink"
    if re.match(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,}$", text) and any(char.isdigit() for char in text):
        return "unit_id_reference"
    return ""
