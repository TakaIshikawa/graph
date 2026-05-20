"""Score retrieved RAG results by factual claim cue density."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from graph.rag.keywords import TOKEN_RE

_MISSING = object()
_ID_KEYS = ("id", "result_id", "unit_id", "source_id")
_TEXT_KEYS = ("claim", "claim_text", "content", "text", "snippet", "summary", "title")

_CUE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("numeric", re.compile(r"(?<![A-Za-z0-9:/.-])(?:\$?\d+(?:\.\d+)?%?|\d{1,3}(?:,\d{3})+)(?![A-Za-z0-9/.-])")),
    ("date", re.compile(r"\b(?:18|19|20)\d{2}(?:-\d{2}-\d{2})?\b")),
    ("comparison", re.compile(r"\b(?:more|less|higher|lower|increased?|decreased?|grew|fell|than|versus|vs)\b", re.IGNORECASE)),
    ("citation", re.compile(r"(?:\[[0-9]+\]|\([A-Z][A-Za-z]+,\s*(?:18|19|20)\d{2}\)|https?://|\bdoi:)", re.IGNORECASE)),
    ("causal", re.compile(r"\b(?:because|therefore|drives|causes|leads to|resulted in|due to)\b", re.IGNORECASE)),
)
_SNIPPET_CHARS = 96


def score_result_claim_density(
    results: Iterable[Any],
    *,
    max_results: int | None = None,
) -> list[dict[str, Any]]:
    """Return per-result factual cue density rows in input order."""
    limit = _validate_max_results(max_results)
    rows = []
    for index, result in enumerate(results):
        if limit is not None and len(rows) >= limit:
            break
        text = _result_text(result)
        token_estimate = max(len(TOKEN_RE.findall(text.casefold())), 1) if text else 0
        cues = _cue_matches(text)
        claim_count = len(cues)
        rows.append(
            {
                "result_id": _result_id(result, index),
                "title": _title(result),
                "claim_count": claim_count,
                "token_estimate": token_estimate,
                "claim_density": round(claim_count / token_estimate, 6) if token_estimate else 0.0,
                "top_cue_snippets": cues,
            }
        )
    return rows


def _validate_max_results(max_results: int | None) -> int | None:
    if max_results is None:
        return None
    if not isinstance(max_results, int) or isinstance(max_results, bool) or max_results < 0:
        raise ValueError("max_results must be a non-negative integer or None")
    return max_results


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


def _title(result: Any) -> str | None:
    return _first_string(result, ("title", "name", "headline"))


def _result_text(result: Any) -> str:
    parts = []
    seen: set[str] = set()
    for key in _TEXT_KEYS:
        text = _first_string(result, (key,))
        if text is not None and text not in seen:
            seen.add(text)
            parts.append(text)
    return " ".join(parts)


def _cue_matches(text: str) -> list[dict[str, str]]:
    rows = []
    seen: set[tuple[str, str]] = set()
    for cue_type, pattern in _CUE_PATTERNS:
        for match in pattern.finditer(text):
            snippet = _snippet(text, match.start(), match.end())
            key = (cue_type, snippet.casefold())
            if key in seen:
                continue
            seen.add(key)
            rows.append({"cue_type": cue_type, "snippet": snippet})
    return sorted(rows, key=lambda row: (row["cue_type"], row["snippet"].casefold()))


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
