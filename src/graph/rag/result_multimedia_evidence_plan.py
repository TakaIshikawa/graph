"""Plan multimedia evidence needs for RAG result sets."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from graph.rag._analysis_utils import content_text, iter_strings, metadata, result_id, string, value

_MEDIA_CUES: dict[str, tuple[tuple[str, re.Pattern[str]], ...]] = {
    "image": (("image", re.compile(r"\b(?:image|photo|picture|screenshot|diagram)\b", re.I)),),
    "audio": (("audio", re.compile(r"\b(?:audio|podcast|recording|sound)\b", re.I)),),
    "video": (("video", re.compile(r"\b(?:video|clip|footage|webinar)\b", re.I)),),
    "chart": (("chart", re.compile(r"\b(?:chart|graph|plot|trendline|visualization)\b", re.I)),),
    "map": (("map", re.compile(r"\b(?:map|geospatial|route|location map)\b", re.I)),),
    "table": (("table", re.compile(r"\b(?:table|spreadsheet|tabular|matrix)\b", re.I)),),
}


def plan_result_multimedia_evidence(
    query: str,
    results: Iterable[Any] = (),
    *,
    expected_format: str | None = None,
) -> dict[str, Any]:
    """Return missing multimedia evidence types and retrieval hints."""
    normalized = _normalize_query(query)
    format_text = _normalize_optional(expected_format)
    required = _required_media(" ".join([normalized, format_text]).strip())
    present_rows = [_result_media(row, index) for index, row in enumerate(results or [])]
    present = sorted({media for row in present_rows for media in row["media_types"]})
    missing = [media for media in required if media not in present]
    return {
        "required_media_types": required,
        "present_media_types": present,
        "missing_media_types": missing,
        "retrieval_hints": [_hint(media) for media in missing],
        "result_media": present_rows,
        "warnings": ["missing_required_media"] if missing else [],
    }


def _required_media(text: str) -> list[str]:
    return [
        media
        for media, cues in _MEDIA_CUES.items()
        if any(pattern.search(text) for _, pattern in cues)
    ]


def _result_media(result: Any, index: int) -> dict[str, Any]:
    meta = metadata(result)
    media_types = set()
    explicit = value(result, "media_type")
    if (text := string(explicit)) and text.casefold() in _MEDIA_CUES:
        media_types.add(text.casefold())
    for key in ("media_types", "attachments", "assets"):
        found = value(result, key)
        if isinstance(found, str):
            candidates = [found]
        elif isinstance(found, Mapping):
            candidates = iter_strings(found)
        else:
            candidates = iter_strings(found)
        for candidate in candidates:
            lowered = candidate.casefold()
            media_types.update(media for media in _MEDIA_CUES if media in lowered)
    text = " ".join([content_text(result), " ".join(iter_strings(meta))])
    media_types.update(media for media, cues in _MEDIA_CUES.items() if any(pattern.search(text) for _, pattern in cues))
    return {"result_id": result_id(result, index), "media_types": sorted(media_types)}


def _hint(media: str) -> str:
    return f"retrieve_{media}_evidence_from_primary_or_specialized_sources"


def _normalize_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    return " ".join(query.casefold().split())


def _normalize_optional(text: str | None) -> str:
    return "" if text is None else " ".join(str(text).casefold().split())
