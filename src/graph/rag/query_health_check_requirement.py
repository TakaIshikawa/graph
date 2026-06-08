"""Detect health-check requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_HEALTH_CHECK_CONTEXT = re.compile(
    r"\b(?:api|app|application|deploy(?:ment)?|endpoint|http|kubernetes|k8s|microservice|probe|service|system)\b",
    re.I,
)
_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("liveness_probe", "high", (r"\bliveness\s+probes?\b", r"\blivez\b", r"\blive\s+checks?\b")),
    ("readiness_probe", "high", (r"\breadiness\s+probes?\b", r"\breadyz\b", r"\bready\s+checks?\b")),
    ("startup_probe", "medium", (r"\bstartup\s+probes?\b", r"\bstartup\s+checks?\b")),
    ("health_endpoint", "medium", (r"\bhealth(?:check)?\s+endpoints?\b", r"/healthz?\b", r"/status\b")),
    (
        "dependency_health",
        "high",
        (
            r"\bdependency\s+health\b",
            r"\bdependencies?\s+(?:health|status|checks?)\b",
            r"\bdependency\s+checks?\b",
            r"\bhealth\s+checks?\s+for\s+dependencies\b",
        ),
    ),
    ("heartbeat", "medium", (r"\bheartbeats?\b", r"\bkeep[-\s]?alive\s+signals?\b")),
)


def detect_query_health_check_requirement(query: str) -> dict[str, Any]:
    """Return health-check requirements mentioned by a query."""
    text = _normalize_query(query)
    requirements = []
    if _HEALTH_CHECK_CONTEXT.search(text):
        for category, severity, patterns in _REQUIREMENTS:
            match = _first_match(patterns, text)
            if match:
                requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})
    requirements.sort(key=lambda row: row["category"])
    return {"has_health_check_requirement": bool(requirements), "requirements": requirements}


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
