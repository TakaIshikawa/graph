"""Pack ranked RAG results into a deterministic context token window."""

from __future__ import annotations

from collections.abc import Sequence, Mapping
from typing import Any

from graph.rag._analysis_utils import content_text, result_id, string
from graph.rag.keywords import TOKEN_RE


def _validate_non_negative_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _estimate_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def _truncate_to_tokens(text: str, budget: int) -> tuple[str, int]:
    if budget <= 0:
        return "", 0
    matches = list(TOKEN_RE.finditer(text))
    if len(matches) <= budget:
        return text, len(matches)
    end = matches[budget - 1].end()
    return text[:end].rstrip(), budget


def plan_context_window_packing(
    results: Sequence[Mapping[str, Any]],
    token_budget: int,
    reserved_tokens: int = 0,
) -> dict[str, Any]:
    """Select full or truncated snippets in input order without exceeding budget."""
    budget = _validate_non_negative_int(token_budget, "token_budget")
    reserved = _validate_non_negative_int(reserved_tokens, "reserved_tokens")
    available = max(0, budget - reserved)
    selected: list[dict[str, Any]] = []
    omitted: list[dict[str, Any]] = []
    used = 0

    for index, result in enumerate(results):
        text = content_text(result)
        estimated = _estimate_tokens(text)
        remaining = available - used
        rid = result_id(result, index)
        title = string(result.get("title")) if isinstance(result, Mapping) else None
        if estimated == 0:
            omitted.append({"result_id": rid, "title": title, "estimated_tokens": 0, "reason": "empty_text"})
            continue
        if estimated <= remaining:
            selected.append(
                {
                    "result_id": rid,
                    "title": title,
                    "estimated_tokens": estimated,
                    "used_tokens": estimated,
                    "selection": "full",
                    "text": text,
                }
            )
            used += estimated
            continue
        if remaining > 0:
            snippet, used_tokens = _truncate_to_tokens(text, remaining)
            selected.append(
                {
                    "result_id": rid,
                    "title": title,
                    "estimated_tokens": estimated,
                    "used_tokens": used_tokens,
                    "selection": "truncated",
                    "text": snippet,
                }
            )
            used += used_tokens
        else:
            omitted.append(
                {
                    "result_id": rid,
                    "title": title,
                    "estimated_tokens": estimated,
                    "reason": "over_budget",
                }
            )

    warnings: list[str] = []
    if available == 0 and results:
        warnings.append("no_available_context_budget")
    if any(item["selection"] == "truncated" for item in selected):
        warnings.append("some_results_truncated")
    if omitted:
        warnings.append("some_results_omitted")

    return {
        "token_budget": budget,
        "reserved_tokens": reserved,
        "available_tokens": available,
        "used_tokens": used,
        "remaining_tokens": available - used,
        "selected": selected,
        "omitted": omitted,
        "warnings": warnings,
    }
