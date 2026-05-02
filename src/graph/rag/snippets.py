"""Build highlighted snippets for RAG/search result payloads."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any


def _validate_snippet_chars(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("snippet_chars must be a non-negative integer")
    return value


def _collapse_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _query_terms(query: Any) -> list[str]:
    if query is None:
        return []

    terms: list[str] = []
    seen: set[str] = set()
    for term in re.findall(r"[\w-]+", str(query).casefold()):
        if term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def _leading_snippet(text: str, snippet_chars: int) -> str:
    if snippet_chars <= 0 or not text:
        return ""
    return text[:snippet_chars].strip()


def _snippet_window(text: str, start: int, snippet_chars: int) -> str:
    if snippet_chars <= 0 or not text:
        return ""
    if len(text) <= snippet_chars:
        return text

    start = max(0, min(start, len(text) - 1))
    end = min(len(text), start + snippet_chars)
    if end == len(text):
        start = max(0, len(text) - snippet_chars)
    return text[start : start + snippet_chars].strip()


def _earliest_match(text: str, terms: list[str]) -> tuple[int, str] | None:
    lowered = text.casefold()
    best: tuple[int, int, str] | None = None
    for index, term in enumerate(terms):
        position = lowered.find(term)
        if position < 0:
            continue
        candidate = (position, index, term)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return None
    return best[0], best[2]


def _highlight_matches(
    snippet: str,
    terms: list[str],
    *,
    marker_start: str,
    marker_end: str,
) -> tuple[str, list[str]]:
    if not snippet or not terms:
        return snippet, []

    lowered = snippet.casefold()
    candidates: list[tuple[int, int, int, str]] = []
    for term_index, term in enumerate(terms):
        for match in re.finditer(re.escape(term), lowered):
            candidates.append((match.start(), match.end(), term_index, term))

    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))
    intervals: list[tuple[int, int, int, str]] = []
    occupied_until = -1
    for start, end, term_index, term in candidates:
        if start < occupied_until:
            continue
        intervals.append((start, end, term_index, term))
        occupied_until = end

    if not intervals:
        return snippet, []

    pieces: list[str] = []
    highlighted_by_key: dict[str, tuple[int, str]] = {}
    cursor = 0
    for start, end, term_index, term in intervals:
        pieces.append(snippet[cursor:start])
        pieces.append(marker_start)
        pieces.append(snippet[start:end])
        pieces.append(marker_end)
        highlighted_by_key.setdefault(term, (term_index, term))
        cursor = end
    pieces.append(snippet[cursor:])

    highlighted_terms = [
        term
        for _term_index, term in sorted(
            highlighted_by_key.values(),
            key=lambda item: item[0],
        )
    ]
    return "".join(pieces), highlighted_terms


def highlight_result_snippets(
    results: Iterable[Mapping[str, Any]],
    query: Any,
    *,
    content_key: str = "content",
    snippet_chars: int = 240,
    marker_start: str = "<mark>",
    marker_end: str = "</mark>",
) -> list[dict[str, Any]]:
    """Return result dictionaries with deterministic snippets and highlights.

    The original result mappings are copied into new dictionaries. Content and
    query whitespace are normalized for matching, but highlighted snippets keep
    the casing from the source content.
    """
    snippet_chars_value = _validate_snippet_chars(snippet_chars)
    terms = _query_terms(query)
    highlighted_results: list[dict[str, Any]] = []

    for result in results:
        payload = dict(result)
        text = _collapse_text(payload.get(content_key))
        first_match = _earliest_match(text, terms)

        if first_match is None:
            snippet = _leading_snippet(text, snippet_chars_value)
        else:
            match_start, _term = first_match
            window_start = max(0, match_start - snippet_chars_value // 3)
            snippet = _snippet_window(text, window_start, snippet_chars_value)

        snippet, highlighted_terms = _highlight_matches(
            snippet,
            terms,
            marker_start=marker_start,
            marker_end=marker_end,
        )
        payload["snippet"] = snippet
        payload["highlighted_terms"] = highlighted_terms
        highlighted_results.append(payload)

    return highlighted_results
