"""Local duplicate candidate ranking for graph units."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from difflib import SequenceMatcher
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from graph.types.models import KnowledgeUnit

TOKEN_RE = re.compile(r"[a-z0-9]+")
URL_KEYS = frozenset(
    {
        "canonical_url",
        "external_url",
        "html_url",
        "htmlurl",
        "link",
        "permalink",
        "source_url",
        "sourceurl",
        "uri",
        "url",
        "web_url",
        "weburl",
        "xml_url",
        "xmlurl",
    }
)
TRACKING_QUERY_PREFIXES = ("utm_",)
TRACKING_QUERY_KEYS = frozenset({"fbclid", "gclid", "mc_cid", "mc_eid", "ref", "ref_src"})
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "with",
    }
)


def _validate_threshold(threshold: float) -> float:
    if not isinstance(threshold, int | float) or isinstance(threshold, bool):
        raise ValueError("threshold must be a number between 0 and 1")
    value = float(threshold)
    if value < 0 or value > 1:
        raise ValueError("threshold must be a number between 0 and 1")
    return value


def _validate_limit(limit: int | None) -> int | None:
    if limit is None:
        return None
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    return limit


def _normalize_text(text: str | None) -> str:
    return " ".join(TOKEN_RE.findall((text or "").lower()))


def _token_set(text: str | None) -> set[str]:
    return {token for token in TOKEN_RE.findall((text or "").lower()) if token not in STOPWORDS}


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    token_containment = 0.0
    if left_tokens and right_tokens:
        token_containment = len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))
    return max(SequenceMatcher(None, left, right).ratio(), token_containment)


def _overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _normalize_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None

    parsed = urlsplit(raw)
    if not parsed.scheme and not parsed.netloc:
        return None

    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    if (scheme == "http" and netloc.endswith(":80")) or (
        scheme == "https" and netloc.endswith(":443")
    ):
        netloc = netloc.rsplit(":", 1)[0]

    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")

    query_items = []
    for key, query_value in parse_qsl(parsed.query, keep_blank_values=True):
        lowered = key.lower()
        if lowered in TRACKING_QUERY_KEYS or lowered.startswith(TRACKING_QUERY_PREFIXES):
            continue
        query_items.append((lowered, query_value))
    query = urlencode(sorted(query_items))

    return urlunsplit((scheme, netloc, path, query, ""))


def _metadata_urls(metadata: Mapping[str, Any]) -> set[str]:
    urls: set[str] = set()
    for key, value in metadata.items():
        normalized_key = str(key).lower().replace("-", "_")
        if normalized_key in URL_KEYS:
            normalized = _normalize_url(value)
            if normalized:
                urls.add(normalized)
    return urls


def _unit_signature(unit: KnowledgeUnit) -> dict[str, Any]:
    title = _normalize_text(unit.title)
    content_tokens = _token_set(unit.content)
    return {
        "id": str(unit.id),
        "source_id": str(unit.source_id).strip(),
        "title": title,
        "content_tokens": content_tokens,
        "urls": _metadata_urls(unit.metadata or {}),
    }


def _score_pair(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any] | None:
    reasons: list[str] = []
    matching_fields: dict[str, Any] = {}
    signal_scores: list[float] = []

    if left["source_id"] and left["source_id"] == right["source_id"]:
        reasons.append("source_id")
        matching_fields["source_id"] = left["source_id"]
        signal_scores.append(0.99)

    matching_urls = sorted(left["urls"] & right["urls"])
    if matching_urls:
        reasons.append("url")
        matching_fields["urls"] = matching_urls
        signal_scores.append(0.98)

    title_similarity = _similarity(left["title"], right["title"])
    if title_similarity:
        matching_fields["title_similarity"] = round(title_similarity, 6)
        if title_similarity >= 0.9:
            reasons.append("title")
            signal_scores.append(0.92 * title_similarity)

    content_overlap = _overlap(left["content_tokens"], right["content_tokens"])
    if content_overlap:
        matching_fields["content_token_overlap"] = round(content_overlap, 6)
        if content_overlap >= 0.75:
            reasons.append("content")
            signal_scores.append(0.9 * content_overlap)

    if title_similarity and content_overlap:
        signal_scores.append((title_similarity * 0.52) + (content_overlap * 0.48))

    if not signal_scores:
        return None

    return {
        "score": round(max(signal_scores), 6),
        "reasons": reasons,
        "matching_fields": matching_fields,
    }


def rank_duplicate_candidates(
    units: Iterable[KnowledgeUnit],
    *,
    threshold: float = 0.82,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic local duplicate candidate pairs above ``threshold``."""
    threshold_value = _validate_threshold(threshold)
    limit_value = _validate_limit(limit)

    signatures = sorted((_unit_signature(unit) for unit in units), key=lambda item: item["id"])
    candidates: list[dict[str, Any]] = []

    for left_index, left in enumerate(signatures):
        for right in signatures[left_index + 1 :]:
            if left["id"] == right["id"]:
                continue
            scored = _score_pair(left, right)
            if not scored or scored["score"] < threshold_value:
                continue
            candidates.append(
                {
                    "unit_ids": [left["id"], right["id"]],
                    "score": scored["score"],
                    "reasons": scored["reasons"],
                    "matching_fields": scored["matching_fields"],
                }
            )

    candidates.sort(key=lambda item: (-item["score"], item["unit_ids"][0], item["unit_ids"][1]))
    if limit_value is not None:
        return candidates[:limit_value]
    return candidates
