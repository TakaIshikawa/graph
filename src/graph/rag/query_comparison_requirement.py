"""Detect comparison intent in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CUE_SPECS: tuple[tuple[str, float, tuple[str, ...]], ...] = (
    ("versus", 0.9, (r"\bversus\b", r"\bvs\.?\b", r"\bagainst\b")),
    ("compare", 0.91, (r"\bcompare\b", r"\bcomparison\b", r"\bcompared\s+(?:with|to)\b")),
    (
        "difference",
        0.86,
        (r"\bdifferences?\s+between\b", r"\bhow\s+do\b.+\bdiffer\b", r"\bdifferentiate\b"),
    ),
    ("preference", 0.84, (r"\bbetter\s+than\b", r"\bwhich\b.+\b(?:better|best)\b")),
    ("tradeoff", 0.84, (r"\btrade[- ]?offs?\b", r"\bpros\s+and\s+cons\b")),
    ("alternatives", 0.78, (r"\balternatives?\b", r"\boptions?\b")),
    ("choice", 0.76, (r"\bshould\s+i\s+(?:use|choose|pick)\b", r"\bchoose\s+between\b")),
    ("or_choice", 0.72, (r"\b[\w.+-]+(?:\s+[\w.+-]+){0,3}\s+or\s+[\w.+-]+(?:\s+[\w.+-]+){0,3}\b",)),
)

_ENTITY_SPLIT_RE = re.compile(
    r"\s+(?:versus|vs\.?|against|compared\s+with|compared\s+to|better\s+than|and|or)\s+",
    re.IGNORECASE,
)
_BOUNDARY_RE = re.compile(r"\s+(?:for|in|when|with|using|under|on|to decide|as)\s+", re.IGNORECASE)
_LEADING_NOISE_RE = re.compile(
    r"^(?:what\s+(?:are|is)\s+|which\s+(?:is|are)\s+|is\s+|are\s+|should\s+i\s+"
    r"(?:use|choose|pick)\s+|compare\s+|the\s+|a\s+|an\s+)",
    re.IGNORECASE,
)
_TRAILING_NOISE_RE = re.compile(r"[\s,.;:?!]+$")


def detect_query_comparison_requirement(query: str) -> dict[str, Any]:
    """Return deterministic comparison intent signals for a query."""
    text = " ".join(str(query or "").split())
    if not text:
        return _result(text, False, "none", [], [], 0.0)

    matches = _cue_matches(text)
    matched_terms = [match["term"] for match in matches]
    requires_comparison = bool(matches)
    comparison_type = _comparison_type(matches)
    entities = _extract_entities(text) if requires_comparison else []
    confidence = _confidence(matches, entities)

    return _result(text, requires_comparison, comparison_type, entities, matched_terms, confidence)


def _result(
    query: str,
    requires_comparison: bool,
    comparison_type: str,
    entities: list[str],
    matched_terms: list[str],
    confidence: float,
) -> dict[str, Any]:
    return {
        "query": query,
        "requires_comparison": requires_comparison,
        "comparison_type": comparison_type,
        "entities": entities,
        "matched_terms": matched_terms,
        "confidence": confidence,
    }


def _cue_matches(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for comparison_type, confidence, patterns in _CUE_SPECS:
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                term = match.group(0).strip()
                key = (comparison_type, term.casefold())
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "comparison_type": comparison_type,
                        "term": term,
                        "confidence": confidence,
                        "start": match.start(),
                    }
                )
    return sorted(rows, key=lambda row: (row["start"], row["comparison_type"], row["term"].casefold()))


def _confidence(matches: list[dict[str, Any]], entities: list[str]) -> float:
    if not matches:
        return 0.0
    confidence = max(float(match["confidence"]) for match in matches)
    if len(matches) > 1:
        confidence += 0.04
    if len(entities) >= 2:
        confidence += 0.04
    return round(min(confidence, 0.98), 2)


def _comparison_type(matches: list[dict[str, Any]]) -> str:
    if not matches:
        return "none"
    selected = min(
        matches,
        key=lambda match: (-float(match["confidence"]), match["start"], match["comparison_type"]),
    )
    return str(selected["comparison_type"])


def _extract_entities(text: str) -> list[str]:
    candidates: list[str] = []
    for segment in _candidate_segments(text):
        for piece in _ENTITY_SPLIT_RE.split(segment):
            cleaned = _clean_entity(piece)
            if cleaned:
                candidates.append(cleaned)
    return _dedupe(candidates)[:6]


def _candidate_segments(text: str) -> list[str]:
    patterns = (
        r"\b(?:compare|comparison\s+of)\s+(.+)",
        r"\b(?:differences?\s+between|choose\s+between)\s+(.+)",
        r"\b(?:pros\s+and\s+cons|trade[- ]?offs?)\s+of\s+(.+)",
        r"\b(.+?\s+better\s+than\s+.+)",
        r"\balternatives?\s+to\s+(.+)",
        r"\b(?:should\s+i\s+(?:use|choose|pick))\s+(.+)",
        r"\bwhich\s+(?:is|are)\s+(?:better|best)[,:]?\s+(.+)",
    )
    segments = [match.group(1) for pattern in patterns for match in re.finditer(pattern, text, re.IGNORECASE)]
    if not segments and _ENTITY_SPLIT_RE.search(text):
        segments.append(text)
    return segments


def _clean_entity(value: str) -> str:
    cleaned = _LEADING_NOISE_RE.sub("", value.strip())
    cleaned = _BOUNDARY_RE.split(cleaned, maxsplit=1)[0]
    cleaned = _TRAILING_NOISE_RE.sub("", cleaned.strip(" '\"()[]{}"))
    if not cleaned or cleaned.casefold() in {
        "better",
        "best",
        "pros",
        "cons",
        "tradeoffs",
        "alternatives",
        "options",
    }:
        return ""
    return cleaned


def _dedupe(values: list[str]) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            rows.append(value)
    return rows
