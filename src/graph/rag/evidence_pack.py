"""Build compact RAG evidence packs from result-like records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.rag.citations import format_result_citations

_MISSING = object()


def build_evidence_pack(
    results: Iterable[Any],
    *,
    limit: int = 10,
    snippet_chars: int = 320,
    summary_char_budget: int = 1200,
) -> dict[str, Any]:
    """Return a deterministic evidence pack for answer generation.

    Inputs may be mappings, ``KnowledgeUnit`` instances, wrappers containing a
    ``unit`` key, or ``(payload, score)`` tuples from ranked search functions.
    """
    limit_value = _validate_non_negative_int(limit, "limit")
    snippet_chars_value = _validate_non_negative_int(snippet_chars, "snippet_chars")
    summary_budget_value = _validate_non_negative_int(
        summary_char_budget,
        "summary_char_budget",
    )

    indexed_results = list(enumerate(results))
    evidence = [
        _evidence_item(result, index, snippet_chars_value)
        for index, result in indexed_results
    ]
    evidence.sort(
        key=lambda item: (
            item["rank"],
            -(item["score"] if item["score"] is not None else -1.0),
            item["title"].casefold(),
            item["id"],
        )
    )
    selected = evidence[:limit_value]
    citations = format_result_citations([item["_raw"] for item in selected])
    for item, citation in zip(selected, citations, strict=False):
        item["citation"] = citation
    for item in selected:
        item.pop("_raw", None)

    source_counts = Counter(item["source_project"] for item in evidence)
    source_counts.pop(None, None)
    summary = _summary_text(selected, summary_budget_value)

    return {
        "total_count": len(evidence),
        "selected_count": len(selected),
        "source_project_counts": dict(sorted(source_counts.items())),
        "source_diversity_count": len(source_counts),
        "evidence": selected,
        "citations": [item["citation"] for item in selected],
        "summary": summary,
        "summary_char_budget": summary_budget_value,
        "truncated": len(summary) < len(_summary_text(selected, None)),
    }


def _validate_non_negative_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _field_value(item: Any, key: str) -> Any:
    if item is _MISSING or item is None:
        return _MISSING
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _payload(result: Any) -> Any:
    if isinstance(result, tuple) and result:
        return result[0]
    return result


def _tuple_score(result: Any) -> Any:
    if isinstance(result, tuple) and len(result) > 1:
        return result[1]
    return _MISSING


def _value(result: Any, key: str) -> Any:
    payload = _payload(result)
    value = _field_value(payload, key)
    if value is not _MISSING and value is not None:
        return value

    metadata = _field_value(payload, "metadata")
    if isinstance(metadata, Mapping):
        metadata_value = metadata.get(key, _MISSING)
        if metadata_value is not _MISSING and metadata_value is not None:
            return metadata_value

    unit = _field_value(payload, "unit")
    if unit is not _MISSING and unit is not None:
        unit_value = _field_value(unit, key)
        if unit_value is not _MISSING and unit_value is not None:
            return unit_value
        unit_metadata = _field_value(unit, "metadata")
        if isinstance(unit_metadata, Mapping):
            metadata_value = unit_metadata.get(key, _MISSING)
            if metadata_value is not _MISSING and metadata_value is not None:
                return metadata_value

    if key == "score":
        return _tuple_score(result)
    return value


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).strip().split())
    return text or None


def _float(value: Any) -> float | None:
    if value is _MISSING or value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _id(result: Any, index: int) -> str:
    for key in ("id", "unit_id", "source_id"):
        value = _string(_value(result, key))
        if value is not None:
            return value
    return f"result-{index + 1}"


def _snippet(result: Any, snippet_chars: int) -> str | None:
    for key in ("snippet", "content", "text", "summary"):
        value = _string(_value(result, key))
        if value is not None:
            return value[:snippet_chars].strip()
    return None


def _evidence_item(result: Any, index: int, snippet_chars: int) -> dict[str, Any]:
    source_project = _string(_value(result, "source_project")) or "unknown"
    return {
        "rank": index + 1,
        "id": _id(result, index),
        "title": _string(_value(result, "title")) or "Untitled",
        "source_project": source_project,
        "source_id": _string(_value(result, "source_id")),
        "source_entity_type": _string(_value(result, "source_entity_type")),
        "snippet": _snippet(result, snippet_chars),
        "score": _float(_value(result, "score")),
        "confidence": _float(_value(result, "confidence")),
        "utility_score": _float(_value(result, "utility_score")),
        "citation": "",
        "_raw": result,
    }


def _summary_text(items: list[dict[str, Any]], budget: int | None) -> str:
    lines = []
    for item in items:
        snippet = item.get("snippet") or ""
        citation = item.get("citation") or ""
        parts = [f"{item['rank']}. {item['title']}"]
        if snippet:
            parts.append(snippet)
        if citation:
            parts.append(citation)
        lines.append(" - ".join(parts))
    text = "\n".join(lines)
    if budget is None or len(text) <= budget:
        return text
    if budget <= 3:
        return text[:budget]
    return text[: budget - 3].rstrip() + "..."
