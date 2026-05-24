"""Cluster context items that likely contradict each other."""

from __future__ import annotations

import re
from collections.abc import Iterable
from itertools import combinations
from typing import Any

from graph.rag._analysis_utils import content_text, ordered_terms, result_id

_CUES: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("trend", "increase", re.compile(r"\b(?:increase[sd]?|higher|grew|growth|rising)\b", re.I)),
    ("trend", "decrease", re.compile(r"\b(?:decrease[sd]?|lower|fell|decline[sd]?|falling)\b", re.I)),
    ("availability", "available", re.compile(r"\b(?:available|enabled|allowed|supported)\b", re.I)),
    ("availability", "unavailable", re.compile(r"\b(?:unavailable|disabled|blocked|unsupported|not supported)\b", re.I)),
)
_OPPOSITE = {("trend", "increase"): "decrease", ("trend", "decrease"): "increase", ("availability", "available"): "unavailable", ("availability", "unavailable"): "available"}


def cluster_context_contradictions(context_items: Iterable[Any]) -> dict[str, Any]:
    """Group context items by shared terms and opposite contradiction cues."""
    rows = [_row(item, index) for index, item in enumerate(context_items)]
    clusters = []
    used: set[str] = set()
    for left, right in combinations(rows, 2):
        shared_terms = sorted(set(left["terms"]) & set(right["terms"]))
        if not shared_terms:
            continue
        triggers = _opposite_triggers(left["cues"], right["cues"])
        if not triggers:
            continue
        ids = sorted([left["item_id"], right["item_id"]])
        used.update(ids)
        clusters.append({"item_ids": ids, "terms": shared_terms[:6], "triggers": triggers})
    return {"clusters": clusters, "unclustered_count": sum(1 for row in rows if row["item_id"] not in used)}


def _row(item: Any, index: int) -> dict[str, Any]:
    text = content_text(item)
    cues = [(group, cue) for group, cue, pattern in _CUES if pattern.search(text)]
    return {"item_id": result_id(item, index), "terms": ordered_terms(text, min_length=4), "cues": cues}


def _opposite_triggers(left: list[tuple[str, str]], right: list[tuple[str, str]]) -> list[str]:
    triggers = []
    for group, cue in left:
        opposite = _OPPOSITE.get((group, cue))
        if opposite and (group, opposite) in right:
            triggers.append(f"{cue}/{opposite}")
    return sorted(set(triggers))
