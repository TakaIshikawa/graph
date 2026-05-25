"""Detect requested output constraints in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def detect_query_output_constraints(query: str) -> dict[str, dict[str, Any]]:
    """Extract normalized answer constraints from a query."""
    text = str(query or "")
    constraints: dict[str, dict[str, Any]] = {}
    _add_number(text, constraints, "word_count", r"\b(?:under|within|in|maximum of|max)?\s*(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+words?\b")
    _add_number(text, constraints, "bullet_count", r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+bullets?\b")
    for key, pattern in {
        "table_format": r"\b(?:as|in)\s+a\s+table\b|\btable format\b",
        "json_format": r"\bjson\b",
        "reading_level": r"\b(?:grade \d+|beginner|expert|plain english)\b",
        "tone": r"\b(?:formal|casual|friendly|concise|neutral) tone\b|\b(?:be concise)\b",
        "deadline": r"\b(?:by|before|due)\s+(?:tomorrow|today|[A-Z][a-z]+\s+\d{1,2})\b",
    }.items():
        match = re.search(pattern, text, re.I)
        if match:
            constraints[key] = {"raw_phrase": match.group(0), "confidence": "high"}
    include = re.search(r"\binclude\s+([^.;]+)", text, re.I)
    if include:
        section_text = re.split(r",\s*(?:formal|casual|friendly|concise|neutral|plain english|grade \d+)\b", include.group(1), flags=re.I)[0]
        constraints["include_sections"] = {"raw_phrase": include.group(0), "sections": _split_sections(section_text), "confidence": "medium"}
    exclude = re.search(r"\b(?:exclude|without)\s+([^.;]+)", text, re.I)
    if exclude:
        section_text = re.split(r"\b(?:by|before|due)\b", exclude.group(1), flags=re.I)[0]
        constraints["exclude_sections"] = {"raw_phrase": exclude.group(0), "sections": _split_sections(section_text), "confidence": "medium"}
    return constraints


def _add_number(text: str, constraints: dict[str, dict[str, Any]], key: str, pattern: str) -> None:
    match = re.search(pattern, text, re.I)
    if not match:
        return
    raw_number = match.group(1).casefold()
    constraints[key] = {"value": int(raw_number) if raw_number.isdigit() else _NUMBER_WORDS[raw_number], "raw_phrase": match.group(0), "confidence": "high"}


def _split_sections(text: str) -> list[str]:
    return [part.strip(" ,") for part in re.split(r",|\band\b", text) if part.strip(" ,")]
