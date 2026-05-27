"""CSV export for likely phone number hints in unit text fields."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "phone_hint", "digit_count", "source_field", "context"]
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d|\(\d)[\d\s().-]{6,}\d(?!\w)")


def export_units_to_phone_number_hint_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows: list[dict[str, str | int]] = []
    for unit in unit_list:
        title = field_value(get(unit, "title") or metadata(unit).get("title"))
        fields = [("title", title), ("content", get(unit, "content"))]
        fields.extend((f"metadata.{key}", value) for key, value in metadata(unit).items() if isinstance(value, str))
        for source, value in fields:
            text = str(value or "")
            for match in _PHONE_RE.finditer(text):
                hint = field_value(match.group(0))
                digits = re.sub(r"\D", "", hint)
                if len(digits) < 7 or len(digits) > 15 or _looks_like_date_or_amount(hint):
                    continue
                rows.append({"unit_id": unit_id(unit), "title": title, "phone_hint": hint, "digit_count": len(digits), "source_field": source, "context": _context(text, match.start(), match.end())})
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), sort_key(row["source_field"]), sort_key(row["phone_hint"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _context(text: str, start: int, end: int) -> str:
    return field_value(text[max(0, start - 24) : min(len(text), end + 24)])


def _looks_like_date_or_amount(value: str) -> bool:
    compact = value.strip()
    return bool(re.fullmatch(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}", compact) or re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d{2})?", compact))
