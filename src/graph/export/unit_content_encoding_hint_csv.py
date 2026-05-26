"""CSV export for content encoding quality hints."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import get, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "content_length", "replacement_char_count", "non_ascii_ratio", "control_char_count", "likely_mojibake", "encoding_note"]
_MOJIBAKE_MARKERS = ("Ã", "Â", "â€™", "â€œ", "â€", "�")


def export_units_to_content_encoding_hint_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = sorted((_row(unit) for unit in unit_list), key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(unit: Mapping[str, Any] | object) -> dict[str, str | int]:
    content = "" if get(unit, "content") is None else str(get(unit, "content"))
    replacement = content.count("\ufffd")
    marker_count = sum(content.count(marker) for marker in _MOJIBAKE_MARKERS)
    non_ascii = sum(1 for char in content if ord(char) > 127)
    controls = sum(1 for char in content if ord(char) < 32 and char not in "\n\r\t")
    notes = []
    if replacement:
        notes.append("replacement_characters")
    if marker_count:
        notes.append("mojibake_markers")
    if controls:
        notes.append("control_characters")
    return {
        "unit_id": unit_id(unit),
        "content_length": len(content),
        "replacement_char_count": replacement,
        "non_ascii_ratio": f"{(non_ascii / len(content)):.4f}" if content else "0.0000",
        "control_char_count": controls,
        "likely_mojibake": str(bool(replacement or marker_count)).lower(),
        "encoding_note": "; ".join(notes),
    }
