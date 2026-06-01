"""Audit whether a RAG answer includes operational fallback paths."""

from __future__ import annotations

import re
from typing import Any

_PATHS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("degraded_mode", (r"\bdegraded\s+mode\b", r"\breduced\s+functionality\b")),
    ("fallback", (r"\bfallback\b", r"\bfailover\b", r"\bcontingenc(?:y|ies)\b")),
    ("manual_workaround", (r"\bmanual\s+workaround\b", r"\bmanual\s+process\b")),
    ("retry", (r"\bretry\b", r"\bretries\b", r"\btry\s+again\b")),
    ("rollback", (r"\brollback\b", r"\broll\s+back\b")),
)
_RECOMMENDED = [name for name, _ in _PATHS]


def audit_answer_fallback_paths(answer: str) -> dict[str, Any]:
    text = " ".join(str(answer or "").split())
    fallback_paths = [name for name, patterns in _PATHS if any(re.search(pattern, text, re.I) for pattern in patterns)]
    return {
        "fallback_paths": fallback_paths,
        "missing_recommended_paths": [name for name in _RECOMMENDED if name not in fallback_paths],
        "has_operational_fallback": bool(fallback_paths),
    }
