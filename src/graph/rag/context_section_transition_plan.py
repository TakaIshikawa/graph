"""Plan answer section transitions from retrieved context records."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import result_date, result_id, source_id, string, tokens, value

_CONFLICT_WORDS = {"however", "but", "contrary", "conflict", "dispute", "although", "whereas"}


def plan_context_section_transitions(context_records: Iterable[Any]) -> dict[str, Any]:
    """Return section boundaries and transition labels for ordered context."""
    records = list(context_records)
    sections: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    previous_features: dict[str, Any] | None = None

    for index, record in enumerate(records):
        features = _features(record, index)
        if current is None or _should_start_section(previous_features, features):
            if current is not None:
                current["end_index"] = index - 1
            current = {
                "section_index": len(sections),
                "start_index": index,
                "end_index": index,
                "record_ids": [],
                "topic": features["topic"],
                "source_type": features["source_type"],
            }
            sections.append(current)
            if previous_features is not None:
                transitions.append(
                    {
                        "from_section": len(sections) - 2,
                        "to_section": len(sections) - 1,
                        "label": _transition_label(previous_features, features),
                        "reason_codes": _reasons(previous_features, features),
                    }
                )
        current["record_ids"].append(features["id"])
        current["end_index"] = index
        previous_features = features
    return {"sections": sections, "transitions": transitions}


def _features(record: Any, index: int) -> dict[str, Any]:
    text = " ".join(filter(None, [string(value(record, "title")), string(value(record, "content")), string(value(record, "text")), string(value(record, "snippet"))]))
    terms = tokens(text, min_length=4)
    topic = string(value(record, "topic")) or (sorted(terms)[0] if terms else "general")
    source_type = string(value(record, "source_type")) or string(value(record, "type")) or source_id(record) or "unknown"
    parsed_date = result_date(record)
    return {
        "id": result_id(record, index),
        "topic": topic.casefold(),
        "terms": terms,
        "source_type": source_type.casefold(),
        "date": parsed_date.isoformat() if parsed_date else None,
        "conflict": bool(terms & _CONFLICT_WORDS),
    }


def _should_start_section(previous: dict[str, Any] | None, current: dict[str, Any]) -> bool:
    if previous is None:
        return True
    return bool(_reasons(previous, current))


def _reasons(previous: dict[str, Any], current: dict[str, Any]) -> list[str]:
    reasons = []
    if previous["topic"] != current["topic"]:
        reasons.append("topic_shift")
    if previous["source_type"] != current["source_type"]:
        reasons.append("source_shift")
    if previous["date"] and current["date"] and previous["date"][:4] != current["date"][:4]:
        reasons.append("date_shift")
    if current["conflict"] and not previous["conflict"]:
        reasons.append("conflict_cue")
    return reasons


def _transition_label(previous: dict[str, Any], current: dict[str, Any]) -> str:
    reasons = _reasons(previous, current)
    if "conflict_cue" in reasons:
        return "contrasting evidence"
    if "date_shift" in reasons:
        return "time shift"
    if "source_shift" in reasons:
        return "source perspective shift"
    return "topic shift"
