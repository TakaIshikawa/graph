"""Audit citation markers in generated RAG answers."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

_MARKER_RE = re.compile(r"\[([^\[\]\n]+)\]")
_ANCHOR_KEYS = ("id", "citation_id", "label", "anchor", "source_id", "title")
_PRIMARY_KEYS = ("id", "citation_id", "label", "anchor", "source_id", "title")


def audit_answer_citation_anchors(answer: str, citations: Iterable[Any]) -> dict[str, Any]:
    """Compare bracket citation markers in an answer to known citation anchors."""
    known_anchors, aliases = _known_anchors(citations)
    used: set[str] = set()
    unknown: set[str] = set()

    for marker in _answer_markers(answer):
        normalized = _normalize(marker)
        canonical = aliases.get(normalized)
        if canonical is None:
            unknown.add(marker)
        else:
            used.add(canonical)

    missing = [anchor for anchor in known_anchors if anchor not in used]
    used_anchors = [anchor for anchor in known_anchors if anchor in used]
    coverage = len(used_anchors) / len(known_anchors) if known_anchors else 0.0

    return {
        "known_anchors": known_anchors,
        "used_anchors": used_anchors,
        "missing_anchors": missing,
        "unknown_anchors": sorted(unknown, key=_sort_key),
        "anchor_coverage": round(coverage, 3),
    }


def _known_anchors(citations: Iterable[Any]) -> tuple[list[str], dict[str, str]]:
    known: list[str] = []
    aliases: dict[str, str] = {}

    for index, citation in enumerate(citations, start=1):
        values = _citation_values(citation)
        canonical = _first_text(values) or str(index)
        if canonical not in known:
            known.append(canonical)
        for value in [str(index), *values]:
            text = _inline_text(value)
            if text:
                aliases[_normalize(text)] = canonical

    return known, aliases


def _citation_values(citation: Any) -> list[str]:
    values: list[str] = []
    for key in _PRIMARY_KEYS:
        text = _inline_text(_get(citation, key))
        if text and text not in values:
            values.append(text)
    metadata = _get(citation, "metadata")
    if isinstance(metadata, Mapping):
        for key in _ANCHOR_KEYS:
            text = _inline_text(metadata.get(key))
            if text and text not in values:
                values.append(text)
    if not values:
        text = _inline_text(citation)
        if text:
            values.append(text)
    return values


def _answer_markers(answer: str) -> list[str]:
    markers: list[str] = []
    for match in _MARKER_RE.finditer(answer or ""):
        text = _inline_text(match.group(1))
        if not text:
            continue
        markers.extend(_split_marker(text))
    return markers


def _split_marker(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"[,;]", text) if part.strip()]
    if len(parts) > 1 and all(part.isdigit() for part in parts):
        return parts
    return [text]


def _first_text(values: Iterable[Any]) -> str:
    for value in values:
        text = _inline_text(value)
        if text:
            return text
    return ""


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _normalize(value: str) -> str:
    return _inline_text(value).casefold()


def _inline_text(value: Any) -> str:
    text = "" if value is None else str(getattr(value, "value", value))
    return " ".join(text.split())


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
