"""Detect privacy-sensitive RAG query requirements."""

from __future__ import annotations

import re
from typing import Any

_SPECS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("pii", re.compile(r"\b(?:pii|personally identifiable|email addresses?|phone numbers?|ssn|social security)\b", re.I)),
    ("private_notes", re.compile(r"\b(?:private notes?|personal notes?|confidential notes?|internal notes?)\b", re.I)),
    ("credentials", re.compile(r"\b(?:credentials?|api keys?|passwords?|tokens?|secrets?)\b", re.I)),
    ("medical", re.compile(r"\b(?:medical|health records?|diagnos(?:is|es)|patient data|hipaa)\b", re.I)),
    ("financial", re.compile(r"\b(?:financial|bank accounts?|credit cards?|tax records?|payroll)\b", re.I)),
    ("legal", re.compile(r"\b(?:legal|attorney[- ]client|privileged|court records?|compliance)\b", re.I)),
    ("anonymization", re.compile(r"\b(?:anonymi[sz]e|de[- ]identify|redact|mask personal|privacy preserving)\b", re.I)),
)


def detect_query_privacy_requirements(query: str) -> dict[str, Any]:
    text = " ".join(("" if query is None else str(query)).split())
    matched = []
    topics = []
    for topic, pattern in _SPECS:
        spans = [{"topic": topic, "text": m.group(0), "start": m.start(), "end": m.end()} for m in pattern.finditer(text)]
        if spans:
            matched.extend(spans)
            topics.append(topic)
    required = bool(matched)
    return {
        "requires_privacy_handling": required,
        "matched_cues": matched,
        "sensitive_topics": topics,
        "confidence": 0.85 if required else 0.0,
        "guidance": "Redact or minimize sensitive data before retrieval and citation." if required else "No special privacy handling detected.",
    }
