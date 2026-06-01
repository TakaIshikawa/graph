"""Summarize Markdown definition-list terms in unit content."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_DEFINITION_RE = re.compile(r"^\s*:\s+(.+)$")


def summarize_unit_markdown_definition_terms(units: Iterable[Mapping[str, Any] | object]) -> dict[str, Any]:
    """Detect definition-list terms and colon-prefixed definitions."""
    unit_list = list(units)
    term_counts: Counter[str] = Counter()
    term_count = definition_count = orphan_definitions = multi_definition_terms = 0
    for unit in unit_list:
        for block in _definition_blocks(_content(unit)):
            if block["orphan"]:
                orphan_definitions += int(block["definitions"])
                continue
            definitions = int(block["definitions"])
            for term in block["terms"]:
                term_text = str(term)
                term_count += 1
                term_counts[term_text] += 1
                definition_count += definitions
                if definitions > 1:
                    multi_definition_terms += 1
    return {
        "total_units": len(unit_list),
        "term_count": term_count,
        "definition_count": definition_count,
        "orphan_definition_count": orphan_definitions,
        "multi_definition_term_count": multi_definition_terms,
        "term_counts": dict(sorted(term_counts.items(), key=lambda item: sort_key(item[0]))),
    }


def _content(unit: Mapping[str, Any] | object) -> str:
    return str(get(unit, "content") or metadata(unit).get("content") or "")


def _definition_blocks(content: str) -> list[dict[str, Any]]:
    lines = _content_lines(content)
    blocks: list[dict[str, Any]] = []
    index = 0
    while index < len(lines):
        line = lines[index][1]
        if _DEFINITION_RE.match(line):
            definitions = 0
            while index < len(lines) and _DEFINITION_RE.match(lines[index][1]):
                definitions += 1
                index += 1
            blocks.append({"terms": [], "definitions": definitions, "orphan": True})
            continue
        terms: list[str] = []
        while index < len(lines):
            candidate = field_value(lines[index][1])
            if not _is_term_candidate(lines[index][1], candidate):
                break
            terms.append(candidate)
            index += 1
        if not terms:
            index += 1
            continue
        while index < len(lines) and not lines[index][1].strip():
            index += 1
        definitions = 0
        while index < len(lines) and _DEFINITION_RE.match(lines[index][1]):
            definitions += 1
            index += 1
        if definitions:
            blocks.append({"terms": terms, "definitions": definitions, "orphan": False})
    return blocks


def _content_lines(content: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            rows.append((line_number, line))
    return rows


def _is_term_candidate(line: str, text: str) -> bool:
    return bool(text) and not line.startswith((" ", "\t")) and "://" not in text and not text.startswith(":")

