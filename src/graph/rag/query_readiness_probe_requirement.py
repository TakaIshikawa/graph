"""Detect readiness probe requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_PROBE_CONTEXT = re.compile(
    r"\b(?:app|application|deploy(?:ment)?|endpoint|health|http|kubernetes|k8s|pod|probe|service|traffic)\b",
    re.I,
)
_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("readiness_probe", "high", (r"\breadiness\s+probes?\b", r"\bready\s+probes?\b", r"\breadyz\b")),
    ("ready_endpoint", "medium", (r"\bready\s+endpoints?\b", r"/readyz?\b", r"/readiness\b")),
    ("traffic_gating", "high", (r"\bgate\s+traffic\b", r"\bremove\s+from\s+service\b", r"\bnot\s+receive\s+traffic\b")),
    ("dependency_readiness", "high", (r"\bdependency\s+readiness\b", r"\bdependencies?\s+(?:are\s+)?ready\b", r"\bready\s+when\s+dependencies?\s+pass\b")),
    ("startup_readiness", "medium", (r"\bready\s+after\s+startup\b", r"\bstartup\s+readiness\b", r"\bmark\s+ready\s+after\s+startup\b")),
    ("failure_threshold", "medium", (r"\bfailure\s+threshold\b", r"\breadiness\s+timeout\b", r"\breadiness\s+period\b")),
)


def detect_query_readiness_probe_requirement(query: str) -> dict[str, Any]:
    """Return readiness probe requirements mentioned by a query."""
    text = _normalize_query(query)
    requirements = []
    if _PROBE_CONTEXT.search(text):
        for category, severity, patterns in _REQUIREMENTS:
            match = _first_match(patterns, text)
            if match:
                requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})
    requirements.sort(key=lambda row: row["category"])
    return {"has_readiness_probe_requirement": bool(requirements), "requirements": requirements}


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
