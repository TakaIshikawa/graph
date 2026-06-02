"""Detect data loss prevention requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("dlp_policy", (r"\bdlp\b", r"\bdata\s+loss\s+prevention\b")),
    ("exfiltration_prevention", (r"\b(?:prevent|block|stop)\s+(?:data\s+)?exfiltration\b", r"\bexfiltration\s+prevention\b")),
    ("sensitive_data_blocking", (r"\bblock(?:ing)?\s+(?:sensitive|confidential|pii)\s+data\b", r"\bsensitive\s+data\s+blocking\b")),
    ("content_inspection", (r"\bcontent\s+inspection\b", r"\binspect\s+(?:uploads?|content|messages?)\b")),
    ("quarantine_workflow", (r"\bquarantine\b", r"\bhold\s+for\s+review\b")),
)
_CHANNELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("email", (r"\bemail\b", r"\bmail\b")),
    ("endpoint", (r"\bendpoint\b", r"\bdevice\b")),
    ("upload", (r"\buploads?\b", r"\bfile\s+transfer\b")),
    ("cloud_storage", (r"\bcloud\s+storage\b", r"\bdrive\b", r"\bs3\b")),
)


def detect_query_data_loss_prevention_requirement(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    cue_categories = _matched(_CUES, text)
    channels = _matched(_CHANNELS, text)
    return {
        "requires_data_loss_prevention": bool(cue_categories),
        "cue_categories": cue_categories,
        "channels": channels if cue_categories else [],
    }


def _matched(specs: tuple[tuple[str, tuple[str, ...]], ...], text: str) -> list[str]:
    return [category for category, patterns in specs if any(re.search(pattern, text, re.I) for pattern in patterns)]


def _normalize_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    return " ".join(query.split())
