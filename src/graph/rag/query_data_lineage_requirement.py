"""Detect data lineage requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_SCOPE_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("data_lineage", (r"\bdata\s+lineage\b", r"\blineage\s+graph\b")),
    ("provenance", (r"\bprovenance\s+chain\b", r"\bdata\s+provenance\b")),
    ("upstream", (r"\bupstream\s+(?:dependencies|sources|datasets?)\b",)),
    ("downstream", (r"\bdownstream\s+(?:dependencies|consumers|outputs?)\b",)),
    ("transformation_history", (r"\btransformation\s+history\b", r"\btrack\s+transformations\b")),
    ("source_to_output", (r"\bsource[-\s]to[-\s]output\s+traceability\b", r"\btrace\s+from\s+source\s+to\s+output\b")),
)


def detect_query_data_lineage_requirement(query: str) -> dict[str, Any]:
    """Return data lineage scopes mentioned by a query."""
    text = " ".join(str(query or "").split())
    scopes = [scope for scope, patterns in _SCOPE_SPECS if any(re.search(pattern, text, re.I) for pattern in patterns)]
    return {
        "requires_data_lineage": bool(scopes),
        "lineage_scopes": scopes,
        "matched_cues": scopes,
        "confidence": "high" if any(scope in {"data_lineage", "source_to_output"} for scope in scopes) else "medium" if scopes else "none",
    }
