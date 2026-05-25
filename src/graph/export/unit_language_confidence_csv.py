"""CSV export for unit language confidence metadata."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, normalized_key, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "detected_language", "declared_language", "confidence", "mismatch_flag", "evidence_fields"]
_DETECTED_KEYS = {"detected_language", "language_detected", "detected_lang"}
_DECLARED_KEYS = {"language", "lang", "declared_language", "locale"}
_CONFIDENCE_KEYS = {"language_confidence", "lang_confidence", "confidence", "detected_language_confidence"}


def export_units_to_language_confidence_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = sorted((_row(unit) for unit in unit_list), key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(unit: Mapping[str, Any] | object) -> dict[str, str]:
    detected, detected_key = _first_value(unit, _DETECTED_KEYS)
    declared, declared_key = _first_value(unit, _DECLARED_KEYS)
    confidence, confidence_key = _first_value(unit, _CONFIDENCE_KEYS)
    evidence = [key for key in (detected_key, declared_key, confidence_key) if key]
    return {
        "unit_id": unit_id(unit),
        "detected_language": detected,
        "declared_language": declared,
        "confidence": confidence,
        "mismatch_flag": "true" if _language_code(detected) and _language_code(declared) and _language_code(detected) != _language_code(declared) else "false",
        "evidence_fields": "; ".join(evidence),
    }


def _first_value(unit: Mapping[str, Any] | object, keys: set[str]) -> tuple[str, str]:
    for key in sorted(keys):
        text = field_value(get(unit, key))
        if text:
            return text, key
    for key, value in metadata(unit).items():
        if normalized_key(key) in keys and field_value(value):
            return field_value(value), f"metadata.{key}"
    return "", ""


def _language_code(value: str) -> str:
    return field_value(value).casefold().replace("_", "-").split("-", 1)[0]
