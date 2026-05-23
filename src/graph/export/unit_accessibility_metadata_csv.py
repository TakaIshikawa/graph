"""CSV export for unit accessibility metadata readiness."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, normalized_key, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "alt_text_present", "transcript_present", "captions_present", "language_present", "aria_label_present", "accessibility_score", "missing_fields"]
_FIELDS = {
    "alt_text": ("alt_text", "alt", "image_alt", "altText"),
    "transcript": ("transcript", "transcription"),
    "captions": ("captions", "caption", "subtitles"),
    "language": ("language", "lang", "locale"),
    "aria_label": ("aria_label", "aria-label", "label"),
}


def export_unit_accessibility_metadata_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write one accessibility metadata audit row per unit."""
    unit_list = list(units)
    rows = [_row(unit) for unit in unit_list]
    rows.sort(key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(unit: Mapping[str, Any] | object) -> dict[str, str]:
    present = {field: _has_value(unit, aliases) for field, aliases in _FIELDS.items()}
    missing = [field for field in _FIELDS if not present[field]]
    return {
        "unit_id": unit_id(unit),
        "alt_text_present": _bool(present["alt_text"]),
        "transcript_present": _bool(present["transcript"]),
        "captions_present": _bool(present["captions"]),
        "language_present": _bool(present["language"]),
        "aria_label_present": _bool(present["aria_label"]),
        "accessibility_score": f"{(sum(present.values()) / len(_FIELDS)):.2f}",
        "missing_fields": "; ".join(missing),
    }


def _has_value(unit: Mapping[str, Any] | object, aliases: tuple[str, ...]) -> bool:
    alias_keys = {normalized_key(alias) for alias in aliases}
    for alias in aliases:
        if field_value(get(unit, alias)):
            return True
    return any(normalized_key(key) in alias_keys and field_value(value) for key, value in metadata(unit).items())


def _bool(value: bool) -> str:
    return "true" if value else "false"
