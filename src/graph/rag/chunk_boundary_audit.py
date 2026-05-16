"""Audit retrieved snippets for suspicious chunk-boundary truncation."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

_MISSING = object()
_TEXT_KEYS = ("content", "text", "snippet")
_OPEN_TO_CLOSE = {"(": ")", "[": "]", "{": "}"}


def _payload(result: Any) -> Any:
    return result[0] if isinstance(result, tuple) and result else result


def _field_value(item: Any, key: str) -> Any:
    if item is _MISSING or item is None:
        return _MISSING
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


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


def _first(result: Any, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        for value in _candidate_values(result, key):
            text = _string(value)
            if text:
                return text
    return None


def _result_id(result: Any, index: int) -> str:
    return _first(result, ("id", "unit_id", "source_id")) or f"result-{index + 1}"


def _content(result: Any) -> str:
    return "\n".join(
        text
        for key in _TEXT_KEYS
        for value in _candidate_values(result, key)
        if (text := _string(value)) is not None
    )


def _unmatched_delimiters(text: str) -> bool:
    stack: list[str] = []
    single_quote = False
    double_quote = False
    for char in text:
        if char == "'" and not double_quote:
            single_quote = not single_quote
        elif char == '"' and not single_quote:
            double_quote = not double_quote
        elif char in _OPEN_TO_CLOSE:
            stack.append(_OPEN_TO_CLOSE[char])
        elif char in _OPEN_TO_CLOSE.values():
            if stack and stack[-1] == char:
                stack.pop()
            else:
                return True
    return bool(stack or single_quote or double_quote)


def _row(result_id: str, issue_type: str, severity: str, text: str) -> dict[str, str]:
    return {
        "result_id": result_id,
        "issue_type": issue_type,
        "severity": severity,
        "evidence": text[:120],
    }


def audit_chunk_boundaries(results: Iterable[Any]) -> dict[str, Any]:
    """Return ordered chunk-boundary issue rows for retrieved snippets."""
    issues: list[dict[str, str]] = []
    result_count = 0
    for index, result in enumerate(results):
        result_count += 1
        result_id = _result_id(result, index)
        content = _content(result)
        stripped = content.strip()
        if not stripped:
            continue
        lower = stripped.casefold()
        if len(stripped) < 35:
            issues.append(_row(result_id, "very-short-fragment", "low", stripped))
        if lower.startswith(("...", "…", "and ", "or ", "but ", "because ", "therefore ", "however ", "continued")) or re.match(r"^[a-z,;:)]", stripped):
            issues.append(_row(result_id, "leading-continuation", "medium", stripped))
        if lower.endswith(("...", "…", " and", " or", " but", " because", " therefore", " however", "continued")) or stripped[-1] in ",;:-":
            issues.append(_row(result_id, "trailing-continuation", "medium", stripped))
        if _unmatched_delimiters(stripped):
            issues.append(_row(result_id, "unmatched-delimiter", "high", stripped))

    severity_order = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda row: (severity_order[row["severity"]], row["result_id"], row["issue_type"]))
    counts = Counter(issue["issue_type"] for issue in issues)
    return {
        "issue_count": len(issues),
        "issues": issues,
        "summary": {"result_count": result_count, "issue_type_counts": dict(sorted(counts.items()))},
    }
