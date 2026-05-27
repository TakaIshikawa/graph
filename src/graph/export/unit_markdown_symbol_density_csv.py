"""CSV export for Markdown structural symbol density by unit."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import get, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "content_length", "markdown_symbol_count", "symbol_density_per_1k_chars", "dominant_symbol"]
_SYMBOLS = "#*_`[]()>|~="
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


def export_units_to_markdown_symbol_density_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = sorted((_row(unit) for unit in unit_list), key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(unit: Mapping[str, Any] | object) -> dict[str, str | int]:
    content = str(get(unit, "content") or "")
    counts: Counter[str] = Counter()
    in_fence = False
    for line in content.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            counts.update(char for char in line if char in _SYMBOLS)
    total = sum(counts.values())
    length = len(content)
    dominant = max(_SYMBOLS, key=lambda symbol: (counts[symbol], -_SYMBOLS.index(symbol))) if total else ""
    return {
        "unit_id": unit_id(unit),
        "content_length": length,
        "markdown_symbol_count": total,
        "symbol_density_per_1k_chars": f"{(total / length * 1000) if length else 0:.2f}",
        "dominant_symbol": dominant,
    }
