"""Score date confidence for extracted RAG evidence claims."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

_MISSING = object()
_ISO_RE = re.compile(r"\b(19\d{2}|20\d{2})-\d{2}-\d{2}\b")
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def _payload(item: Any) -> Any:
    return item[0] if isinstance(item, tuple) and item else item


def _field_value(item: Any, key: str) -> Any:
    if item is _MISSING or item is None:
        return _MISSING
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _candidate_values(item: Any, key: str):
    payload = _payload(item)
    value = _field_value(payload, key)
    if value is not _MISSING:
        yield value
    metadata = _field_value(payload, "metadata")
    if isinstance(metadata, Mapping):
        value = metadata.get(key, _MISSING)
        if value is not _MISSING:
            yield value


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def _first(item: Any, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        for value in _candidate_values(item, key):
            if (text := _string(value)):
                return text
    return None


def _id(item: Any, index: int) -> str:
    return _first(item, ("id", "claim_id", "source_id")) or f"claim-{index + 1}"


def _years(values: Iterable[str]) -> set[str]:
    years: set[str] = set()
    for value in values:
        years.update(_YEAR_RE.findall(value))
    return years


def score_claim_date_confidence(claims: Iterable[Any]) -> dict[str, Any]:
    """Classify claims as dated, inferred-date, undated, or conflicting-date."""
    rows: list[dict[str, Any]] = []
    for index, claim in enumerate(claims):
        text = _first(claim, ("claim", "text", "content")) or ""
        text_dates = _ISO_RE.findall(text)
        text_years = _years([text])
        metadata_values = [
            value
            for key in ("date", "published_at", "publication_date", "created_at", "updated_at")
            for raw in _candidate_values(claim, key)
            if (value := _string(raw)) is not None
        ]
        metadata_years = _years(metadata_values)
        metadata_dates = [match.group(0) for value in metadata_values for match in _ISO_RE.finditer(value)]
        if text_years and metadata_years and text_years.isdisjoint(metadata_years):
            label = "conflicting-date"
            confidence = 0.25
        elif text_dates or text_years:
            label = "dated"
            confidence = 1.0 if text_dates else 0.85
        elif metadata_dates or metadata_years:
            label = "inferred-date"
            confidence = 0.65
        else:
            label = "undated"
            confidence = 0.0
        rows.append(
            {
                "claim_id": _id(claim, index),
                "label": label,
                "confidence": confidence,
                "claim_dates": sorted(set(match.group(0) for match in _ISO_RE.finditer(text))),
                "claim_years": sorted(text_years),
                "metadata_dates": sorted(set(metadata_dates)),
                "metadata_years": sorted(metadata_years),
            }
        )

    counts = Counter(row["label"] for row in rows)
    return {"claim_count": len(rows), "claims": rows, "summary": dict(sorted(counts.items()))}
