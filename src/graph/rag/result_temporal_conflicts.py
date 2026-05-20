"""Detect temporal status conflicts across retrieved RAG results."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from datetime import date
from typing import Any

from graph.rag._analysis_utils import result_date, result_id, string, tokens, value

_TEXT_KEYS = ("topic", "claim", "title", "snippet", "summary", "content", "text")
_STATUS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("unavailable", re.compile(r"\bunavailable|not\s+available\b", re.IGNORECASE)),
    ("available", re.compile(r"\bavailable\b", re.IGNORECASE)),
    ("deprecated", re.compile(r"\bdeprecated|retired|sunset(?:ted)?\b", re.IGNORECASE)),
    ("active", re.compile(r"\bactive|current(?:ly)?\s+supported\b", re.IGNORECASE)),
    ("completed", re.compile(r"\bcompleted|complete|finished|launched|released\b", re.IGNORECASE)),
    ("planned", re.compile(r"\bplanned|proposed|scheduled|roadmap\b", re.IGNORECASE)),
    ("closed", re.compile(r"\bclosed|resolved|fixed\b", re.IGNORECASE)),
    ("open", re.compile(r"\bopen|unresolved|pending\b", re.IGNORECASE)),
    ("decreased", re.compile(r"\bdecreased|declined|fell|reduced\b", re.IGNORECASE)),
    ("increased", re.compile(r"\bincreased|rose|grew|expanded\b", re.IGNORECASE)),
)
_CONFLICT_PAIRS = {
    frozenset(("planned", "completed")): "planned_vs_completed",
    frozenset(("active", "deprecated")): "active_vs_deprecated",
    frozenset(("available", "unavailable")): "available_vs_unavailable",
    frozenset(("open", "closed")): "open_vs_closed",
    frozenset(("increased", "decreased")): "increased_vs_decreased",
}
_STATUS_WORDS = {
    "active",
    "available",
    "closed",
    "complete",
    "completed",
    "current",
    "decreased",
    "deprecated",
    "expanded",
    "fell",
    "finished",
    "fixed",
    "grew",
    "increased",
    "launched",
    "open",
    "pending",
    "planned",
    "proposed",
    "reduced",
    "released",
    "resolved",
    "retired",
    "roadmap",
    "rose",
    "scheduled",
    "sunset",
    "sunsetted",
    "unavailable",
    "unresolved",
}


def detect_result_temporal_conflicts(results: Iterable[Any]) -> list[dict[str, Any]]:
    """Return older/newer result pairs with conflicting temporal status cues."""
    try:
        rows = list(results or [])
    except TypeError:
        rows = []

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, result in enumerate(rows):
        text = _text(result)
        status = _status(result, text)
        dated = result_date(result)
        topic = _topic(text)
        if status is None or dated is None or topic is None:
            continue
        grouped[topic].append(
            {
                "result": result,
                "index": index,
                "date": dated,
                "status": status,
            }
        )

    conflicts = []
    for topic in sorted(grouped):
        items = sorted(grouped[topic], key=lambda item: (item["date"], result_id(item["result"], item["index"])))
        for older_index, older in enumerate(items):
            for newer in items[older_index + 1 :]:
                conflict_type = _conflict_type(older["status"], newer["status"])
                if conflict_type is None:
                    continue
                conflicts.append(_row(topic, older, newer, conflict_type))
    return conflicts


def _text(result: Any) -> str:
    return " ".join(text for key in _TEXT_KEYS if (text := string(value(result, key))) is not None)


def _status(result: Any, text: str) -> str | None:
    explicit = string(value(result, "status")) or string(value(result, "state"))
    haystack = " ".join(part for part in (explicit, text) if part)
    for label, pattern in _STATUS_PATTERNS:
        if pattern.search(haystack):
            return label
    return None


def _topic(text: str) -> str | None:
    terms = sorted(term for term in tokens(text, min_length=2) if term not in _STATUS_WORDS)
    return " ".join(terms) if terms else None


def _conflict_type(first: str, second: str) -> str | None:
    return _CONFLICT_PAIRS.get(frozenset((first, second)))


def _row(topic: str, older: dict[str, Any], newer: dict[str, Any], conflict_type: str) -> dict[str, Any]:
    older_date: date = older["date"]
    newer_date: date = newer["date"]
    return {
        "topic": topic,
        "older_result_id": result_id(older["result"], older["index"]),
        "newer_result_id": result_id(newer["result"], newer["index"]),
        "older_date": older_date.isoformat(),
        "newer_date": newer_date.isoformat(),
        "conflict_type": conflict_type,
        "reason": f"{older['status']} status on older evidence conflicts with {newer['status']} status on newer evidence",
    }
