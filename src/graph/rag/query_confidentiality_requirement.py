"""Detect confidentiality requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_SIGNALS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("confidential", re.compile(r"\bconfidential(?:ity)?\b", re.I)),
    ("nda", re.compile(r"\bnda\b|non[-\s]?disclosure", re.I)),
    ("internal_only", re.compile(r"\binternal\s+only\b|company[-\s]?internal", re.I)),
    ("private_dataset", re.compile(r"\bprivate\s+(?:data(?:set)?|corpus|records?)\b", re.I)),
    ("do_not_share", re.compile(r"\bdo\s+not\s+share\b|\bdon't\s+share\b|not\s+for\s+sharing", re.I)),
    ("anonymize", re.compile(r"\banonymi[sz]e\b|de[-\s]?identify", re.I)),
    ("redact", re.compile(r"\bredact(?:ion|ed)?\b", re.I)),
)
_REDACTION_SIGNALS = {"do_not_share", "anonymize", "redact"}


def detect_query_confidentiality_requirement(query: str) -> dict[str, Any]:
    text = " ".join(str(query or "").split())
    signals = [name for name, pattern in _SIGNALS if pattern.search(text)]
    return {
        "requires_confidentiality": bool(signals),
        "signals": signals,
        "redaction_recommended": any(signal in _REDACTION_SIGNALS for signal in signals),
    }
