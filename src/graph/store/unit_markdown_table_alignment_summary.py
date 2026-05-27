"""Summarize Markdown pipe table alignment markers."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


def summarize_unit_markdown_table_alignments(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    total = units_with = tables = columns = malformed = 0
    alignments: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for index, unit in enumerate(units):
        total += 1
        uid = unit_id(unit) or str(index)
        found = False
        lines = _content_lines(unit)
        for pos in range(len(lines) - 1):
            line_no, delimiter = lines[pos + 1]
            if "|" not in lines[pos][1] or "|" not in delimiter:
                continue
            cells = [cell.strip() for cell in delimiter.strip().strip("|").split("|")]
            if len(cells) < 2 or not all(_looks_delimiter(cell) for cell in cells):
                if any("-" in cell for cell in cells):
                    malformed += 1
                continue
            sequence = [_alignment(cell) for cell in cells]
            found = True
            tables += 1
            columns += len(sequence)
            alignments.update(sequence)
            if len(examples) < sample_limit:
                examples.append({"unit_id": uid, "line": line_no, "alignment_sequence": sequence, "column_count": len(sequence)})
        if found:
            units_with += 1
    return {"total_units": total, "units_with_tables": units_with, "total_tables": tables, "total_columns": columns, "alignment_counts": dict(sorted(alignments.items())), "malformed_delimiter_rows": malformed, "examples": examples}


def _looks_delimiter(cell: str) -> bool:
    return bool(cell) and "-" in cell and set(cell) <= {"-", ":"}


def _alignment(cell: str) -> str:
    return "center" if cell.startswith(":") and cell.endswith(":") else "left" if cell.startswith(":") else "right" if cell.endswith(":") else "unspecified"


def _content_lines(unit: Any) -> list[tuple[int, str]]:
    in_fence = False
    rows = []
    for line_no, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            rows.append((line_no, line))
    return rows
