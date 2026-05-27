"""Detect answer output format requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_FORMAT_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("table", ("table", "tabular")),
    ("json", ("json",)),
    ("checklist", ("checklist", "check list")),
    ("bullet_list", ("bullet list", "bullets", "bullet points")),
    ("timeline", ("timeline", "chronology")),
    ("csv", ("csv", "comma separated", "comma-separated")),
    ("step_by_step", ("step-by-step", "step by step")),
)
_INTENT_RE = re.compile(r"\b(?:return|give|provide|format|as|in|must be|prefer|use|write|make)\b")
_STRICT_RE = re.compile(r"\b(?:must|only|exactly|strict|valid|required|return\s+json|must\s+be)\b")
_PREFER_RE = re.compile(r"\b(?:prefer|ideally|if possible|can be)\b")


def detect_query_output_format_requirements(query: str) -> dict[str, Any]:
    normalized_query = " ".join(str(query or "").casefold().split())
    cues = _cues(normalized_query)
    formats = list(dict.fromkeys(cue["format"] for cue in cues))
    return {
        "has_format_requirement": bool(formats),
        "formats": formats,
        "primary_format": formats[0] if formats else None,
        "strictness": _strictness(normalized_query, formats),
        "matched_cues": cues,
        "normalized_query": normalized_query,
    }


def _cues(query: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    has_global_intent = _INTENT_RE.search(query) is not None
    for format_name, cues in _FORMAT_SPECS:
        for cue in cues:
            for match in re.finditer(rf"\b{re.escape(cue)}\b", query):
                context = query[max(0, match.start() - 24) : min(len(query), match.end() + 24)]
                if has_global_intent or _INTENT_RE.search(context):
                    rows.append({"format": format_name, "cue": cue, "span": [match.start(), match.end()]})
                    break
    rows.sort(key=lambda row: (row["span"][0], row["span"][1], row["format"]))
    return rows


def _strictness(query: str, formats: list[str]) -> str:
    if not formats:
        return "none"
    if _STRICT_RE.search(query):
        return "strict"
    if _PREFER_RE.search(query):
        return "preferred"
    return "requested"
