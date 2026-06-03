"""Detect tenant isolation requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_SAAS_CONTEXT_RE = re.compile(r"\b(?:multi[-\s]?tenant|tenant\s+(?:data|boundary|isolation|workspace|account|org|organization)|saas|customer\s+(?:data|environment|account)|cross[-\s]?tenant)\b", re.I)
_LEGAL_TENANT_RE = re.compile(r"\b(?:lease|landlord|rent|apartment|property|eviction)\b", re.I)
_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("compute_isolation", "medium", (r"\bcompute\s+isolation\b", r"\bisolated\s+(?:workers?|runtime|containers?|compute)\b", r"\bper[-\s]?tenant\s+(?:workers?|runtime|containers?|compute)\b")),
    ("cross_tenant_access", "high", (r"\bcross[-\s]?tenant\s+access\b", r"\btenant[-\s]?to[-\s]?tenant\s+access\b", r"\baccess\s+across\s+tenants\b")),
    ("data_isolation", "high", (r"\btenant\s+data\s+isolation\b", r"\bdata\s+isolation\b", r"\bisolate\s+(?:each\s+)?tenant\s+data\b", r"\bper[-\s]?tenant\s+(?:database|schema|storage)\b")),
    ("network_isolation", "medium", (r"\bnetwork\s+isolation\b", r"\bper[-\s]?tenant\s+(?:vpcs?|networks?)\b", r"\btenant\s+vpcs?\b")),
    ("noisy_neighbor", "medium", (r"\bnoisy\s+neighbor\b", r"\btenant\s+resource\s+(?:contention|limits?)\b", r"\bper[-\s]?tenant\s+(?:quotas?|rate\s+limits?)\b")),
    ("tenant_boundary", "high", (r"\btenant\s+boundar(?:y|ies)\b", r"\borganization\s+boundar(?:y|ies)\b", r"\bcustomer\s+boundar(?:y|ies)\b")),
)


def detect_query_tenant_isolation_requirements(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    has_context = bool(_SAAS_CONTEXT_RE.search(text)) and not bool(_LEGAL_TENANT_RE.search(text))
    requirements = []
    if has_context:
        for category, severity, patterns in _REQUIREMENTS:
            match = _first_match(patterns, text)
            if match:
                requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})
    requirements.sort(key=lambda row: row["category"])
    return {"has_tenant_isolation_requirements": bool(requirements), "requirements": requirements}


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
