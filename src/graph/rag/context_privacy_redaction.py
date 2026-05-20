"""Plan privacy redactions for retrieved RAG context without mutating content."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, result_id

_FINDERS = (
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "high"),
    ("phone", re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"), "high"),
    ("api_key", re.compile(r"\b(?:sk|pk|api|key|token)[-_]?[A-Za-z0-9]{16,}\b", re.I), "high"),
    ("credit_card", re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "high"),
    ("precise_location", re.compile(r"\b\d{1,6}\s+[A-Z][A-Za-z0-9.-]+(?:\s+[A-Z][A-Za-z0-9.-]+){0,5}\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Drive|Dr)\b"), "low"),
)


def plan_context_privacy_redactions(results: Iterable[Any], *, include_low_confidence: bool = False) -> dict[str, Any]:
    """Return redaction findings with masked previews only."""
    rows = []
    for index, result in enumerate(results):
        findings = []
        for redaction_type, pattern, confidence in _FINDERS:
            if confidence == "low" and not include_low_confidence:
                continue
            for match in pattern.finditer(content_text(result)):
                raw = match.group(0)
                if redaction_type == "credit_card" and not _card_like(raw):
                    continue
                findings.append(
                    {
                        "redaction_type": redaction_type,
                        "confidence": confidence,
                        "masked_preview": _mask(raw),
                        "action": f"redact_{redaction_type}",
                    }
                )
        if findings:
            rows.append({"result_id": result_id(result, index), "findings": findings, "finding_count": len(findings)})
    counts = Counter(finding["redaction_type"] for row in rows for finding in row["findings"])
    return {
        "result_count": len(rows),
        "finding_count": sum(row["finding_count"] for row in rows),
        "redaction_type_counts": {kind: counts.get(kind, 0) for kind, _, _ in _FINDERS},
        "results": rows,
        "warnings": ["sensitive_context_detected"] if rows else [],
    }


def _card_like(text: str) -> bool:
    digits = re.sub(r"\D", "", text)
    return 13 <= len(digits) <= 19


def _mask(text: str) -> str:
    compact = " ".join(text.split())
    if len(compact) <= 8:
        return "*" * len(compact)
    return f"{compact[:3]}...{compact[-3:]}"
