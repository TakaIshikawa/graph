"""Extract action items from retrieved RAG/search results."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

_MISSING = object()
_TEXT_KEYS = ("content", "text", "snippet")
_ID_KEYS = ("id", "unit_id", "source_id")
_CHECKBOX_RE = re.compile(r"^\s*[-*+]\s+\[[ xX]\]\s+")
_PREFIX_CUE_RE = re.compile(r"^\s*(?:[-*+]\s*)?(TODO|FIXME|action item|follow up|next step)\s*:?\s*", re.IGNORECASE)
_INLINE_CUE_RE = re.compile(r"\b(action item|follow up|next step)\s*:?\s*", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")


def _payload(result: Any) -> Any:
    return result[0] if isinstance(result, tuple) and result else result


def _field_value(item: Any, key: str) -> Any:
    if item is _MISSING or item is None:
        return _MISSING
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _candidate_values(result: Any, key: str) -> Iterable[Any]:
    payload = _payload(result)
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


def _first(result: Any, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        for value in _candidate_values(result, key):
            if (text := _string(value)):
                return text
    return None


def _result_id(result: Any, index: int) -> str:
    return _first(result, _ID_KEYS) or f"result-{index + 1}"


def _raw_content(result: Any) -> str:
    return "\n".join(
        str(value)
        for key in _TEXT_KEYS
        for value in _candidate_values(result, key)
        if value is not _MISSING and value is not None and str(value).strip()
    )


def _clean_action(text: str) -> str | None:
    cleaned = _CHECKBOX_RE.sub("", text)
    cleaned = _PREFIX_CUE_RE.sub("", cleaned)
    cleaned = " ".join(cleaned.split()).strip(" -:")
    return cleaned or None


def _cue_label(value: str) -> str:
    return " ".join(value.casefold().split())


def _iter_actions(text: str) -> Iterable[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    for line in text.splitlines():
        if _CHECKBOX_RE.match(line):
            action = _clean_action(line)
            if action is not None:
                key = ("checkbox", action)
                if key not in seen:
                    seen.add(key)
                    yield key
            continue
        if match := _PREFIX_CUE_RE.match(line):
            action = _clean_action(line)
            if action is not None:
                key = (_cue_label(match.group(1)), action)
                if key not in seen:
                    seen.add(key)
                    yield key

    for sentence_match in _SENTENCE_RE.finditer(text):
        sentence = sentence_match.group(0)
        if _CHECKBOX_RE.match(sentence) or _PREFIX_CUE_RE.match(sentence):
            continue
        if match := _INLINE_CUE_RE.search(sentence):
            action = " ".join(sentence[match.end() :].split()).strip(" -:.")
            if action:
                key = (_cue_label(match.group(1)), action)
                if key not in seen:
                    seen.add(key)
                    yield key


def extract_result_action_items(results: Iterable[Any], *, max_items: int = 20) -> list[dict[str, Any]]:
    """Return deterministic action items found in retrieved result text."""
    if not isinstance(max_items, int) or isinstance(max_items, bool) or max_items < 0:
        raise ValueError("max_items must be a non-negative integer")
    if max_items == 0:
        return []

    rows: list[dict[str, Any]] = []
    for result_index, result in enumerate(results):
        content = _raw_content(result)
        if not content:
            continue
        result_id = _result_id(result, result_index)
        for cue, action in _iter_actions(content):
            rows.append(
                {
                    "result_id": result_id,
                    "action": action,
                    "cue": cue,
                    "position": len(rows),
                }
            )
            if len(rows) >= max_items:
                return rows
    return rows
