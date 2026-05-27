"""Audit evidence license compatibility."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

_DEFAULT_ALLOWED = {"cc0", "cc-by", "cc-by-sa", "mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "public-domain"}
_LICENSE_KEYS = ("license", "rights", "usage_rights")


def audit_evidence_license_compatibility(evidence: Iterable[Mapping[str, Any]], allowed_licenses: Iterable[str] | None = None) -> dict[str, Any]:
    allowed = {_normalize(item) for item in (allowed_licenses or _DEFAULT_ALLOWED)}
    total = compatible = incompatible = missing = unknown = 0
    ids = {"compatible": [], "incompatible": [], "missing": [], "unknown": []}
    for index, item in enumerate(evidence):
        total += 1
        evidence_id = str(item.get("id") or item.get("evidence_id") or index)
        raw = _license(item)
        if raw is None or str(raw).strip() == "":
            missing += 1
            ids["missing"].append(evidence_id)
            continue
        normalized = _normalize(raw)
        if normalized in allowed:
            compatible += 1
            ids["compatible"].append(evidence_id)
        elif normalized in {"unknown", "unspecified", "n/a"}:
            unknown += 1
            ids["unknown"].append(evidence_id)
        else:
            incompatible += 1
            ids["incompatible"].append(evidence_id)
    ratio = compatible / total if total else 0.0
    return {
        "total_evidence": total,
        "compatible_count": compatible,
        "incompatible_count": incompatible,
        "missing_count": missing,
        "unknown_count": unknown,
        "compatible_ratio": ratio,
        "compatible_evidence_ids": ids["compatible"],
        "incompatible_evidence_ids": ids["incompatible"],
        "missing_evidence_ids": ids["missing"],
        "unknown_evidence_ids": ids["unknown"],
    }


def _license(item: Mapping[str, Any]) -> Any:
    for key in _LICENSE_KEYS:
        if key in item:
            return item[key]
    meta = item.get("metadata")
    if isinstance(meta, Mapping):
        for key in _LICENSE_KEYS:
            if key in meta:
                return meta[key]
    return None


def _normalize(value: Any) -> str:
    return str(value).strip().casefold().replace("_", "-").replace(" ", "-")
