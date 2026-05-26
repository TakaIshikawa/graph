"""CSV export for quote-like content in units."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, flatten_values, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "quote_line_count", "quote_block_count", "metadata_quote_count", "longest_quote_chars", "has_attribution_marker"]
_QUOTE_KEYS = {"quote", "quotes", "excerpt", "excerpts", "highlight", "highlights"}
_ATTRIBUTION_KEYS = {"author", "source", "citation", "attribution"}


def export_units_to_quote_inventory_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = sorted((_row(unit) for unit in unit_list), key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(unit: Mapping[str, Any] | object) -> dict[str, str | int]:
    content = "" if get(unit, "content") is None else str(get(unit, "content"))
    quote_lines = [_quote_text(line) for line in content.splitlines() if _quote_text(line)]
    metadata_quotes = [field_value(value) for key, raw in metadata(unit).items() if field_value(key).casefold() in _QUOTE_KEYS for value in flatten_values(raw) if field_value(value)]
    longest = max([len(text) for text in quote_lines + metadata_quotes], default=0)
    return {
        "unit_id": unit_id(unit),
        "quote_line_count": len(quote_lines),
        "quote_block_count": _quote_block_count(content),
        "metadata_quote_count": len(metadata_quotes),
        "longest_quote_chars": longest,
        "has_attribution_marker": "true" if _has_attribution(content, unit) else "false",
    }


def _quote_text(line: str) -> str:
    stripped = line.lstrip()
    if not stripped.startswith(">"):
        return ""
    return field_value(stripped.lstrip("> "))


def _quote_block_count(content: str) -> int:
    count = 0
    in_block = False
    for line in content.splitlines():
        is_quote = bool(_quote_text(line))
        if is_quote and not in_block:
            count += 1
        in_block = is_quote
    return count


def _has_attribution(content: str, unit: Mapping[str, Any] | object) -> bool:
    if any(field_value(key).casefold() in _ATTRIBUTION_KEYS and field_value(value) for key, value in metadata(unit).items()):
        return True
    return any(line.lstrip().startswith(("> -", "> --", "> —")) for line in content.splitlines())
