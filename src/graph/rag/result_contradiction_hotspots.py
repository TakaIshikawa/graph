"""Analyze contradiction hotspots across RAG results."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, source_id, string, value

_CUES = ("contradicts", "conflicts", "disputes", "retracted", "inconsistent")


def analyze_result_contradiction_hotspots(results: Iterable[Any]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"result_count": 0, "contradiction_cue_count": 0, "matched_cues": set()})
    for result in results or []:
        key = _group_key(result)
        groups[key]["result_count"] += 1
        text = content_text(result).casefold()
        for cue in _CUES:
            count = len(re.findall(rf"\b{re.escape(cue)}\b", text))
            if count:
                groups[key]["contradiction_cue_count"] += count
                groups[key]["matched_cues"].add(cue)
    rows = []
    for key, data in groups.items():
        count = data["contradiction_cue_count"]
        severity = "high" if count >= 2 else "medium" if count else "none"
        rows.append({"group_key": key, "result_count": data["result_count"], "contradiction_cue_count": count, "matched_cues": sorted(data["matched_cues"]), "severity": severity})
    rank = {"high": 0, "medium": 1, "none": 2}
    return sorted(rows, key=lambda row: (rank[row["severity"]], row["group_key"]))


def _group_key(result: Any) -> str:
    for key in ("entity", "topic", "title"):
        text = string(value(result, key))
        if text:
            return re.sub(r"\W+", "-", text.casefold()).strip("-")
    return (source_id(result) or "unknown").casefold()
