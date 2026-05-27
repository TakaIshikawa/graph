"""Analyze redundant claims in result records."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import iter_strings, result_id, value

_CLAIM_KEYS = ("claim", "claims", "summary", "title", "content")


def analyze_result_claim_redundancy(results: Iterable[Any]) -> dict[str, Any]:
    claims = []
    for index, result in enumerate(results):
        rid = result_id(result, index)
        for key in _CLAIM_KEYS:
            for text in iter_strings(value(result, key)):
                if text:
                    claims.append({"result_id": rid, "claim": text, "tokens": _tokens(text)})
    groups: list[list[dict[str, Any]]] = []
    for claim in claims:
        for group in groups:
            if _jaccard(claim["tokens"], group[0]["tokens"]) >= 0.75:
                group.append(claim)
                break
        else:
            groups.append([claim])
    redundant = [
        {"claims": [c["claim"] for c in group], "result_ids": [c["result_id"] for c in group], "normalized_claim": _normalize(group[0]["claim"])}
        for group in groups
        if len(group) > 1
    ]
    return {"redundant_groups": redundant, "unique_claim_count": len(groups), "claim_count": len(claims)}


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"\b[a-z0-9]+\b", text.casefold()) if len(token) > 2}


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left or right else 0.0


def _normalize(text: str) -> str:
    return " ".join(sorted(_tokens(text)))
