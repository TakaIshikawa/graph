"""Audit whether an answer contains a compact risk register."""

from __future__ import annotations

import re
from typing import Any

REQUIRED_FIELDS = ("risk", "likelihood", "impact", "mitigation", "owner", "status")

_ALIASES: dict[str, tuple[str, ...]] = {
    "risk": ("risk", "issue", "threat"),
    "likelihood": ("likelihood", "probability", "chance"),
    "impact": ("impact", "severity", "effect"),
    "mitigation": ("mitigation", "mitigate", "response", "action"),
    "owner": ("owner", "assignee", "responsible", "lead"),
    "status": ("status", "state", "stage"),
}
_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(?P<body>.+?)\s*$")
_LABEL_RE = re.compile(r"(?:^|[;|,]\s*)(?P<label>[A-Za-z][A-Za-z _-]{1,30})\s*:\s*")


def audit_answer_risk_register(answer: str) -> dict[str, Any]:
    """Return required-field coverage and detected item count for risk registers."""
    text = "" if answer is None else str(answer)
    table_fields, table_rows = _table_signals(text)
    bullet_fields, bullet_rows = _bullet_signals(text)
    present = [field for field in REQUIRED_FIELDS if field in table_fields or field in bullet_fields]
    missing = [field for field in REQUIRED_FIELDS if field not in present]
    item_count = table_rows + bullet_rows

    return {
        "has_risk_register": "risk" in present and item_count > 0,
        "present_fields": present,
        "missing_fields": missing,
        "risk_item_count": item_count,
    }


def _table_signals(text: str) -> tuple[set[str], int]:
    fields: set[str] = set()
    row_count = 0
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if "|" not in line:
            continue
        cells = _table_cells(line)
        mapped = {_field_for(cell) for cell in cells}
        mapped.discard(None)
        if "risk" not in mapped:
            continue
        fields.update(mapped)
        row_count += _count_table_rows(lines, index)
    return fields, row_count


def _count_table_rows(lines: list[str], header_index: int) -> int:
    count = 0
    for line in lines[header_index + 1 :]:
        if "|" not in line:
            break
        cells = _table_cells(line)
        if cells and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells):
            continue
        if any(cell.strip() for cell in cells):
            count += 1
    return count


def _table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _bullet_signals(text: str) -> tuple[set[str], int]:
    fields: set[str] = set()
    count = 0
    for line in text.splitlines():
        match = _BULLET_RE.match(line)
        if not match:
            continue
        labels = {_field_for(label.group("label")) for label in _LABEL_RE.finditer(match.group("body"))}
        labels.discard(None)
        if "risk" in labels:
            count += 1
            fields.update(labels)
    return fields, count


def _field_for(label: str) -> str | None:
    normalized = re.sub(r"[^a-z]+", " ", label.casefold()).strip()
    for field, aliases in _ALIASES.items():
        if normalized in aliases:
            return field
    return None
