"""Analyze version conflicts across RAG evidence items."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, result_id, string, value

_VERSION_FIELDS = ("version", "api_version", "product_version", "package_version", "doc_version", "release", "release_version")
_VERSION_RE = re.compile(r"\b(?:v(?:ersion)?\s*)?\d+(?:\.\d+){1,3}(?:[-+][A-Za-z0-9.-]+)?\b", re.I)


def analyze_evidence_version_conflicts(evidence_items: Iterable[Any]) -> dict[str, Any]:
    """Return grouped version evidence and conflict flags."""
    items = list(evidence_items or [])
    groups: dict[str, dict[str, Any]] = {}
    versioned_count = 0
    for index, item in enumerate(items):
        evidence_id = result_id(item, index)
        versions = _versions_for(item)
        if versions:
            versioned_count += 1
        for version in versions:
            group = groups.setdefault(version, {"version": version, "evidence_ids": [], "count": 0})
            group["evidence_ids"].append(evidence_id)
            group["count"] += 1
    version_groups = sorted(groups.values(), key=lambda row: row["version"])
    for group in version_groups:
        group["evidence_ids"] = sorted(group["evidence_ids"])
    conflict_count = 1 if len(version_groups) > 1 else 0
    return {
        "total_evidence": len(items),
        "versioned_evidence_count": versioned_count,
        "version_groups": version_groups,
        "conflict_count": conflict_count,
        "conflict_flags": _conflict_flags(version_groups),
    }


def _versions_for(item: Any) -> list[str]:
    seen: set[str] = set()
    versions: list[str] = []
    for field in _VERSION_FIELDS:
        text = string(value(item, field))
        if text is None:
            continue
        for version in _extract_versions(text):
            if version not in seen:
                seen.add(version)
                versions.append(version)
    for version in _extract_versions(content_text(item)):
        if version not in seen:
            seen.add(version)
            versions.append(version)
    return versions


def _extract_versions(text: str) -> list[str]:
    versions = []
    for match in _VERSION_RE.finditer(text):
        version = _normalize_version(match.group(0))
        if version not in versions:
            versions.append(version)
    return versions


def _normalize_version(version: str) -> str:
    normalized = " ".join(version.strip().split()).casefold()
    normalized = re.sub(r"^version\s+", "", normalized)
    normalized = re.sub(r"^v(?=\d)", "", normalized)
    return normalized


def _conflict_flags(version_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(version_groups) < 2:
        return []
    return [
        {
            "type": "multiple_versions",
            "versions": [group["version"] for group in version_groups],
            "evidence_ids": sorted({evidence_id for group in version_groups for evidence_id in group["evidence_ids"]}),
        }
    ]
