"""Analyze media type mix across retrieved RAG results."""

from __future__ import annotations

from typing import Any

from graph.rag._analysis_utils import string, value

_KEYS = ("media_type", "content_type", "mime_type", "type")


def analyze_result_media_type_mix(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Return normalized media type counts."""
    total = len(results or [])
    counts: dict[str, int] = {}
    for item in results or []:
        media_type = _normalize(_raw_type(item))
        counts[media_type] = counts.get(media_type, 0) + 1
    dominant = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0] if counts else None
    return {
        "total_results": total,
        "media_type_counts": dict(sorted(counts.items())),
        "unknown_count": counts.get("unknown", 0),
        "dominant_media_type": dominant,
        "diverse_media_types": len([key for key in counts if key != "unknown"]) > 1,
    }


def _raw_type(item: Any) -> str | None:
    for key in _KEYS:
        text = string(value(item, key))
        if text:
            return text
    return None


def _normalize(raw: str | None) -> str:
    text = (raw or "").casefold()
    if not text:
        return "unknown"
    if "pdf" in text:
        return "pdf"
    if "video" in text:
        return "video"
    if "audio" in text:
        return "audio"
    if "image" in text:
        return "image"
    if "dataset" in text or "csv" in text or "json" in text:
        return "dataset"
    if "html" in text or "article" in text or "text/" in text or text in {"web", "page"}:
        return "article"
    return "unknown"
