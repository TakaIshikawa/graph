"""Flag likely conflicts across retrieved RAG context snippets."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from graph.rag.keywords import COMMON_STOPWORDS, TOKEN_RE

_MISSING = object()
_ID_KEYS = ("id", "result_id", "unit_id", "source_id")
_TEXT_KEYS = ("claim", "claim_text", "snippet", "content", "text", "summary", "title")
_SNIPPET_CHARS = 150
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9:/.-])\$?(-?\d+(?:\.\d+)?)(%?)(?![A-Za-z0-9/.-])")

_CUE_GROUPS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("trend", "increase", re.compile(r"\b(?:increase[sd]?|increasing|grew|growth|rose|rising|higher)\b", re.IGNORECASE)),
    ("trend", "decrease", re.compile(r"\b(?:decrease[sd]?|decreasing|fell|falling|decline[sd]?|lower)\b", re.IGNORECASE)),
    ("status", "enabled", re.compile(r"\b(?:enabled|active|available|on|allowed)\b", re.IGNORECASE)),
    ("status", "disabled", re.compile(r"\b(?:disabled|inactive|unavailable|off|blocked)\b", re.IGNORECASE)),
    ("support", "supported", re.compile(r"\b(?:supported|supports|confirmed|valid|works)\b", re.IGNORECASE)),
    ("support", "unsupported", re.compile(r"\b(?:unsupported|not supported|invalid|fails|does not work)\b", re.IGNORECASE)),
    ("sequence", "before", re.compile(r"\bbefore\b", re.IGNORECASE)),
    ("sequence", "after", re.compile(r"\bafter\b", re.IGNORECASE)),
)
_OPPOSITES = {
    ("trend", "increase"): "decrease",
    ("trend", "decrease"): "increase",
    ("status", "enabled"): "disabled",
    ("status", "disabled"): "enabled",
    ("support", "supported"): "unsupported",
    ("support", "unsupported"): "supported",
    ("sequence", "before"): "after",
    ("sequence", "after"): "before",
}


def flag_context_conflicts(query: str, results: Iterable[Any]) -> dict[str, Any]:
    """Return deterministic likely-conflict flags for shared query terms."""
    query_terms = _query_terms(query)
    analyzed = [_analyze_result(result, index, query_terms) for index, result in enumerate(results)]
    flags = _cue_conflicts(analyzed) + _numeric_conflicts(analyzed)
    flags.sort(key=lambda row: (row["term"], row["conflict_type"], row["cue_pair"], row["result_ids"]))
    return {"query_terms": query_terms, "conflict_count": len(flags), "flags": flags}


def _query_terms(query: str) -> list[str]:
    terms = {
        token
        for token in TOKEN_RE.findall(str(query).casefold())
        if len(token) >= 3 and token not in COMMON_STOPWORDS
    }
    return sorted(terms)


def _analyze_result(result: Any, index: int, query_terms: list[str]) -> dict[str, Any]:
    text = _result_text(result)
    folded = text.casefold()
    matched_terms = [term for term in query_terms if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", folded)]
    cues = [
        {"group": group, "cue": cue, "snippet": _snippet(text, match.start(), match.end())}
        for group, cue, pattern in _CUE_GROUPS
        for match in pattern.finditer(text)
    ]
    numbers = [
        {
            "value": float(match.group(1)),
            "display": f"{match.group(1)}{match.group(2)}",
            "snippet": _snippet(text, match.start(), match.end()),
        }
        for match in _NUMBER_RE.finditer(text)
    ]
    return {
        "result_id": _result_id(result, index),
        "text": text,
        "matched_terms": matched_terms,
        "cues": cues,
        "numbers": numbers,
    }


def _cue_conflicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flags = []
    seen: set[tuple[str, str, tuple[str, str]]] = set()
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            shared_terms = sorted(set(left["matched_terms"]) & set(right["matched_terms"]))
            if not shared_terms:
                continue
            for left_cue in left["cues"]:
                opposite = _OPPOSITES.get((left_cue["group"], left_cue["cue"]))
                if opposite is None:
                    continue
                for right_cue in right["cues"]:
                    if right_cue["group"] != left_cue["group"] or right_cue["cue"] != opposite:
                        continue
                    cue_pair = " / ".join(sorted([left_cue["cue"], right_cue["cue"]]))
                    result_ids = tuple(sorted([left["result_id"], right["result_id"]]))
                    for term in shared_terms:
                        key = (term, cue_pair, result_ids)
                        if key in seen:
                            continue
                        seen.add(key)
                        flags.append(
                            {
                                "term": term,
                                "conflict_type": "opposing_cues",
                                "cue_pair": cue_pair,
                                "result_ids": list(result_ids),
                                "confidence": "high",
                                "evidence": _evidence(left, left_cue["snippet"], right, right_cue["snippet"]),
                            }
                        )
    return flags


def _numeric_conflicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flags = []
    by_term: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for term in row["matched_terms"]:
            if row["numbers"]:
                by_term[term].append(row)
    for term, term_rows in by_term.items():
        for left_index, left in enumerate(term_rows):
            for right in term_rows[left_index + 1 :]:
                for left_number in left["numbers"]:
                    for right_number in right["numbers"]:
                        if left_number["value"] == right_number["value"]:
                            continue
                        values = sorted([left_number["display"], right_number["display"]])
                        flags.append(
                            {
                                "term": term,
                                "conflict_type": "numeric_disagreement",
                                "cue_pair": "numeric disagreement",
                                "result_ids": sorted([left["result_id"], right["result_id"]]),
                                "confidence": "medium",
                                "values": values,
                                "evidence": _evidence(left, left_number["snippet"], right, right_number["snippet"]),
                            }
                        )
                        break
                    else:
                        continue
                    break
    return flags


def _payload(result: Any) -> Any:
    if isinstance(result, tuple) and result:
        return result[0]
    return result


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
    unit = _field_value(payload, "unit")
    if unit is not _MISSING and unit is not None:
        value = _field_value(unit, key)
        if value is not _MISSING:
            yield value
        metadata = _field_value(unit, "metadata")
        if isinstance(metadata, Mapping):
            value = metadata.get(key, _MISSING)
            if value is not _MISSING:
                yield value


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).strip().split())
    return text or None


def _first_string(result: Any, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        for value in _candidate_values(result, key):
            text = _string(value)
            if text is not None:
                return text
    return None


def _result_id(result: Any, index: int) -> str:
    return _first_string(result, _ID_KEYS) or f"result-{index + 1}"


def _result_text(result: Any) -> str:
    parts = []
    seen: set[str] = set()
    for key in _TEXT_KEYS:
        text = _first_string(result, (key,))
        if text is not None and text not in seen:
            seen.add(text)
            parts.append(text)
    return " ".join(parts)


def _snippet(text: str, start: int, end: int) -> str:
    half = max((_SNIPPET_CHARS - (end - start)) // 2, 12)
    left = max(start - half, 0)
    right = min(end + half, len(text))
    snippet = text[left:right].strip()
    if left > 0:
        snippet = f"...{snippet}"
    if right < len(text):
        snippet = f"{snippet}..."
    return snippet


def _evidence(left: dict[str, Any], left_snippet: str, right: dict[str, Any], right_snippet: str) -> list[dict[str, str]]:
    return sorted(
        [
            {"result_id": left["result_id"], "snippet": left_snippet},
            {"result_id": right["result_id"], "snippet": right_snippet},
        ],
        key=lambda row: row["result_id"],
    )
