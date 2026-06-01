"""Detect incident response requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_SIGNAL_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("incident_response", (r"\bincident\s+response\b", r"\bir\s+plan\b", r"\bincident\s+playbook\b")),
    ("triage", (r"\btriage\b", r"\binitial\s+assessment\b", r"\bprioriti[sz]e\s+the\s+incident\b")),
    ("severity", (r"\bsev(?:erity)?[-\s]?[0-5]\b", r"\bseverity\s+(?:level|classification|rating)\b")),
    ("containment", (r"\bcontain(?:ment)?\b", r"\bisolat(?:e|ion)\b", r"\bquarantine\b")),
    ("remediation", (r"\bremediat(?:e|ion)\b", r"\bmitigat(?:e|ion)\b", r"\bfix\s+forward\b")),
    ("postmortem", (r"\bpost[-\s]?mortem\b", r"\bpost[-\s]?incident\s+review\b", r"\blessons\s+learned\b")),
    ("communications", (r"\bcustomer\s+communications?\b", r"\bnotify\s+(?:customers|users)\b", r"\bcomms\s+plan\b")),
    ("status_page", (r"\bstatus\s+page\b", r"\bpublic\s+status\b", r"\bservice\s+status\s+update\b")),
)

_CUSTOMER_COMMUNICATION_SIGNALS = {"communications", "status_page"}


def detect_query_incident_response_requirement(query: str) -> dict[str, Any]:
    """Return incident response requirement signals mentioned by a query."""
    text = " ".join(str(query or "").split())
    signals = [
        signal
        for signal, patterns in _SIGNAL_SPECS
        if any(re.search(pattern, text, re.I) for pattern in patterns)
    ]
    return {
        "requires_incident_response": bool(signals),
        "signals": signals,
        "customer_communication_required": any(signal in _CUSTOMER_COMMUNICATION_SIGNALS for signal in signals),
    }
