"""Summarize orphaned Markdown footnote references and definitions."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_DEF_RE = re.compile(r"^\s*\[\^(?P<label>[^\]]+)\]:")
_REF_RE = re.compile(r"\[\^(?P<label>[^\]]+)\]")


def summarize_unit_footnote_orphans(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    total_units = 0
    unresolved_total = 0
    unused_total = 0
    examples = []
    affected = set()
    for unit in units:
        total_units += 1
        refs: dict[str, list[int]] = {}
        defs: dict[str, int] = {}
        for line_number, line in _markdown_lines(unit):
            def_match = _DEF_RE.match(line)
            if def_match:
                defs[field_value(def_match.group("label"))] = line_number
                continue
            for ref in _REF_RE.finditer(line):
                refs.setdefault(field_value(ref.group("label")), []).append(line_number)
        unresolved = [(label, line) for label, lines in refs.items() if label not in defs for line in lines]
        unused = [(label, line) for label, line in defs.items() if label not in refs]
        unresolved_total += len(unresolved)
        unused_total += len(unused)
        if unresolved or unused:
            affected.add(unit_id(unit))
        for label, line in unresolved:
            if len(examples) < sample_limit:
                examples.append({"unit_id": unit_id(unit), "line": line, "label": label, "orphan_type": "unresolved_reference"})
        for label, line in unused:
            if len(examples) < sample_limit:
                examples.append({"unit_id": unit_id(unit), "line": line, "label": label, "orphan_type": "unused_definition"})
    examples.sort(key=lambda row: (sort_key(row["unit_id"]), row["line"], sort_key(row["label"]), sort_key(row["orphan_type"])))
    return {"total_units": total_units, "unresolved_references": unresolved_total, "unused_definitions": unused_total, "affected_units": len(affected), "examples": examples}


def _markdown_lines(unit: Any) -> list[tuple[int, str]]:
    rows = []
    in_fence = False
    for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if not in_fence:
            rows.append((line_number, line))
    return rows
