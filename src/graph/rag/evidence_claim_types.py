"""Classify evidence snippets into conservative claim type buckets."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, result_id

_SIGNALS = {
    "numeric": (r"\b\d+(?:\.\d+)?%?\b", r"\b(?:million|billion|percent|average|median)\b"),
    "comparative": (r"\b(?:more|less|higher|lower|better|worse|versus|compared|than)\b",),
    "causal": (r"\b(?:because|therefore|due to|caused|drives|leads to|resulted in)\b",),
    "definitional": (r"\b(?:is defined as|refers to|means|is a type of)\b",),
    "normative": (r"\b(?:should|must|recommended|best practice|required)\b",),
    "procedural": (r"\b(?:step|first|then|submit|install|configure|process)\b",),
    "anecdotal": (r"\b(?:case study|interview|participant said|reported that|story)\b",),
    "factual": (r"\b(?:is|are|was|were|has|have|occurred|changed|reported)\b",),
}


def classify_evidence_claim_types(snippets: Iterable[Any]) -> list[dict[str, Any]]:
    """Classify each evidence snippet into zero or more claim types."""
    rows = []
    for index, snippet in enumerate(snippets or []):
        text = content_text(snippet) or str(snippet or "")
        matches = []
        for claim_type, patterns in _SIGNALS.items():
            signals = sorted({match.group(0) for pattern in patterns for match in re.finditer(pattern, text, re.I)}, key=str.casefold)
            if signals:
                matches.append({"claim_type": claim_type, "signals": signals})
        confidence = "low" if not matches or (len(matches) == 1 and matches[0]["claim_type"] == "factual") else "medium"
        if any(match["claim_type"] in {"numeric", "causal", "definitional"} for match in matches):
            confidence = "high"
        rows.append({"id": result_id(snippet, index), "claim_types": [match["claim_type"] for match in matches], "matched_signals": matches, "confidence": confidence})
    return rows
