"""Extract entity-like focus terms from RAG queries."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

_QUOTED_RE = re.compile(r'"([^"]+)"|“([^”]+)”|\'([^\']+)\'')
_URL_RE = re.compile(r"\bhttps?://[^\s)]+|\b(?:[a-z0-9-]+\.)+[a-z]{2,}\b", re.IGNORECASE)
_HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{2,}")
_TAG_RE = re.compile(r"(?<!\w)#[A-Za-z0-9_][A-Za-z0-9_-]*")
_CAP_RE = re.compile(r"\b(?:[A-Z][\w&'.-]+(?:\s+|$)){2,}")


def extract_query_entity_focus(query: str) -> list[dict[str, Any]]:
    """Return normalized entity-like focus entries in first-seen order."""
    text = str(query or "")
    candidates: list[dict[str, Any]] = []
    occupied_spans: list[tuple[int, int]] = []
    for pattern_type, pattern in (("quoted_phrase", _QUOTED_RE), ("url_or_domain", _URL_RE), ("handle", _HANDLE_RE), ("hashtag", _TAG_RE)):
        for match in pattern.finditer(text):
            original = next((group for group in match.groups() if group), match.group(0))
            candidates.append(_entry(pattern_type, original, match.start()))
            occupied_spans.append(match.span())
    for match in _CAP_RE.finditer(text):
        if any(match.start() < end and match.end() > start for start, end in occupied_spans):
            continue
        original = " ".join(match.group(0).strip(" ,.;:!?()[]{}").split())
        if len(original.split()) >= 2:
            candidates.append(_entry("capitalized_name", original, match.start()))
    candidates.sort(key=lambda item: item["position"])
    seen = set()
    rows = []
    for item in candidates:
        key = (item["type"], item["text"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(item)
    return rows


def _entry(kind: str, original: str, position: int) -> dict[str, Any]:
    return {"type": kind, "text": _normalize(kind, original), "original_text": original, "position": position}


def _normalize(kind: str, text: str) -> str:
    cleaned = " ".join(text.strip().strip(".,;:!?").split())
    if kind == "url_or_domain":
        parsed = urlparse(cleaned if "://" in cleaned else f"https://{cleaned}")
        host = parsed.netloc.casefold()
        if host.startswith("www."):
            host = host[4:]
        return host or cleaned.casefold()
    if kind in {"handle", "hashtag"}:
        return cleaned.casefold()
    return cleaned.casefold()
