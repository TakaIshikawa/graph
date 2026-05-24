"""CSV export for lightweight citation style hints in units."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, flatten_values, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "detected_styles", "citation_count", "has_doi", "has_url_reference", "evidence"]
_TEXT_KEYS = {"citation", "citations", "reference", "references", "bibliography", "doi", "url", "urls", "links"}
_APA_RE = re.compile(r"\([A-Z][A-Za-z' -]+,\s*(?:19|20)\d{2}[a-z]?\)")
_NUMERIC_RE = re.compile(r"\[(?:\d{1,3}(?:\s*,\s*\d{1,3})*|\d{1,3}\s*-\s*\d{1,3})\]")
_DOI_RE = re.compile(r"(?:doi:\s*|https?://(?:dx\.)?doi\.org/)?10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
_FOOTNOTE_RE = re.compile(r"(?:\[\^[^\]]+\]|\b\d+\s*$)")
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


def export_unit_citation_style_inventory_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write per-unit citation style detections."""
    unit_list = list(units)
    rows = sorted((_row(unit) for unit in unit_list), key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(unit: Mapping[str, Any] | object) -> dict[str, str | int]:
    text = _unit_text(unit)
    detections = _detections(text)
    styles = [name for name in ("apa_parenthetical", "numeric_bracket", "doi", "footnote", "url_reference") if detections[name]]
    evidence = [detections[name][0] for name in styles]
    return {
        "unit_id": unit_id(unit),
        "detected_styles": "; ".join(styles),
        "citation_count": sum(len(values) for values in detections.values()),
        "has_doi": "true" if detections["doi"] else "false",
        "has_url_reference": "true" if detections["url_reference"] else "false",
        "evidence": "; ".join(evidence[:5]),
    }


def _unit_text(unit: Mapping[str, Any] | object) -> str:
    parts = [field_value(get(unit, "content")), field_value(get(unit, "title"))]
    for key, value in metadata(unit).items():
        if field_value(key).casefold().replace("-", "_") in _TEXT_KEYS:
            parts.extend(field_value(item) for item in flatten_values(value))
    return "\n".join(part for part in parts if part)


def _detections(text: str) -> dict[str, list[str]]:
    urls = _URL_RE.findall(text)
    return {
        "apa_parenthetical": _APA_RE.findall(text),
        "numeric_bracket": _NUMERIC_RE.findall(text),
        "doi": _DOI_RE.findall(text),
        "footnote": _FOOTNOTE_RE.findall(text),
        "url_reference": urls,
    }
