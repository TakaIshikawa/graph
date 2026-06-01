"""Detect environment constraints in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_ENVIRONMENT_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("dev", (r"\bdev(?:elopment)?\b", r"\bdeveloper\s+environment\b")),
    ("staging", (r"\bstaging\b", r"\bstage\s+env(?:ironment)?\b", r"\bpre[-\s]?prod(?:uction)?\b", r"\buat\b")),
    ("production", (r"\bprod(?:uction)?\b", r"\blive\s+(?:site|service|environment|traffic)\b")),
    ("sandbox", (r"\bsandbox(?:ed)?\b", r"\btest\s+sandbox\b")),
    ("air_gapped_lab", (r"\bair[-\s]?gapped\s+lab\b", r"\boffline\s+lab\b", r"\bisolated\s+lab\b")),
    (
        "region_specific",
        (
            r"\b(?:eu|us|uk|apac|emea|latam|jp|japan|canada|australia)[-\s](?:region|environment|env)\b",
            r"\bregion[-\s]specific\b",
            r"\bin\s+(?:the\s+)?(?:eu|us|uk|apac|emea|latam|japan|canada|australia)\s+(?:region|environment|env)\b",
        ),
    ),
    ("local", (r"\blocal(?:host)?\b", r"\blocal\s+(?:env(?:ironment)?|machine|development)\b", r"\bon\s+my\s+machine\b")),
)

_PRODUCTION = "production"


def detect_query_environment_constraints(query: str) -> dict[str, Any]:
    """Return normalized environment constraints mentioned by a query."""
    text = " ".join(str(query or "").split())
    environments = [
        environment
        for environment, patterns in _ENVIRONMENT_SPECS
        if any(re.search(pattern, text, re.I) for pattern in patterns)
    ]
    return {
        "has_environment_constraints": bool(environments),
        "environments": environments,
        "production_sensitive": _PRODUCTION in environments,
    }
