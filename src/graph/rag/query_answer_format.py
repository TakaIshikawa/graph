"""Plan answer format from query wording."""

from __future__ import annotations

import re
from typing import Any

_FORMATS = (
    ("json", ("json", "machine readable", "schema")),
    ("table", ("table", "columns", "spreadsheet")),
    ("checklist", ("checklist", "check list", "todo", "to-do")),
    ("timeline", ("timeline", "chronology", "chronological", "over time")),
    ("comparison", ("compare", "comparison", "versus", " vs ", "pros and cons")),
    ("step_by_step", ("step-by-step", "step by step", "steps", "how to")),
    ("summary", ("summary", "summarize", "brief", "overview")),
    ("citations", ("cite", "citations", "sources", "with links")),
)


def plan_query_answer_format(query: str) -> dict[str, Any]:
    """Infer a normalized response format plan from query cues."""
    text = " ".join(str(query or "").split())
    lowered = f" {text.casefold()} "
    formats = [label for label, cues in _FORMATS if any(_contains(lowered, cue) for cue in cues)]
    if not formats:
        formats = ["summary"]
    sections = _sections(formats)
    warnings = []
    if "json" in formats and any(fmt in formats for fmt in ("table", "checklist", "timeline", "step_by_step")):
        warnings.append("conflicting_structured_format_requests")
    return {"formats": formats, "sections": sections, "ordering_hints": _ordering(formats), "warnings": warnings}


def _sections(formats: list[str]) -> list[str]:
    if "comparison" in formats:
        return ["criteria", "options", "tradeoffs", "recommendation"]
    if "timeline" in formats:
        return ["date", "event", "evidence"]
    if "checklist" in formats:
        return ["items", "status"]
    return ["answer"]


def _ordering(formats: list[str]) -> list[str]:
    if "timeline" in formats:
        return ["chronological"]
    if "comparison" in formats:
        return ["group_by_option", "then_by_criterion"]
    if "step_by_step" in formats:
        return ["sequential"]
    return []


def _contains(text: str, cue: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(cue)}(?!\w)", text))
