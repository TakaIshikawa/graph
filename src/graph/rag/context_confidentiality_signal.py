"""Analyze retrieved context for confidentiality signals."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._record_text import record_id, text_blob

_SIGNALS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("confidential", "high", re.compile(r"\bconfidential(?:ity)?\b", re.I)),
    ("nda", "high", re.compile(r"\bnda\b|non[-\s]?disclosure", re.I)),
    ("internal_only", "medium", re.compile(r"\binternal[-\s]only\b|\bcompany[-\s]?internal\b|\binternal\s+use\s+only\b", re.I)),
    ("customer_data", "high", re.compile(r"\bcustomer\s+(?:data|records?|information|account)\b", re.I)),
    ("private_key", "critical", re.compile(r"\bprivate\s+key\b|-----BEGIN\s+(?:RSA\s+|EC\s+|OPENSSH\s+)?PRIVATE\s+KEY-----", re.I)),
    ("token", "critical", re.compile(r"\b(?:api[-_\s]?token|access[-_\s]?token|bearer\s+token|token\s*[:=]\s*[A-Za-z0-9._-]{12,})\b", re.I)),
    ("secret", "critical", re.compile(r"\b(?:client[-_\s]?secret|shared\s+secret|secret\s*[:=]\s*[A-Za-z0-9._-]{8,})\b", re.I)),
    ("do_not_share", "high", re.compile(r"\bdo\s+not\s+share\b|\bdon't\s+share\b|\bnot\s+for\s+(?:external\s+)?sharing\b", re.I)),
)
_SEVERITY_ORDER = {"medium": 1, "high": 2, "critical": 3}


def analyze_context_confidentiality_signals(context_items: Iterable[Any]) -> dict[str, Any]:
    """Return confidentiality findings for each flagged retrieved context item."""
    rows = []
    counts: Counter[str] = Counter()
    for index, item in enumerate(context_items or []):
        text = text_blob(item)
        findings = []
        for signal, severity, pattern in _SIGNALS:
            match = pattern.search(text)
            if match:
                counts[signal] += 1
                findings.append({"signal": signal, "matched_text": match.group(0), "severity": severity})
        if findings:
            max_severity = max((finding["severity"] for finding in findings), key=lambda severity: _SEVERITY_ORDER[severity])
            rows.append(
                {
                    "context_id": record_id(item, index, prefix="context"),
                    "severity": max_severity,
                    "findings": findings,
                    "finding_count": len(findings),
                }
            )
    return {
        "flagged_context_item_count": len(rows),
        "finding_count": sum(row["finding_count"] for row in rows),
        "signal_counts": {signal: counts.get(signal, 0) for signal, _, _ in _SIGNALS},
        "items": rows,
        "confidentiality_review_recommended": bool(rows),
    }
