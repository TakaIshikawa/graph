"""Summarize attachment checksum coverage and duplicates."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

_ATTACHMENT_KEYS = ("attachments", "files", "documents")
_CHECKSUM_KEYS = ("checksum", "sha256", "sha1", "md5")
_ALGORITHM_KEYS = ("checksum_algorithm", "algorithm", "hash_algorithm")
_UNIT_ID_KEYS = ("id", "unit_id")


def summarize_attachment_checksums(units: Iterable[Any]) -> dict[str, Any]:
    """Aggregate attachment checksum coverage, algorithms, and duplicates."""

    total = checksummed = missing = 0
    algorithms: Counter[str] = Counter()
    checksum_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for unit in units:
        unit_id = _unit_id(unit)
        for index, attachment in enumerate(_attachments(unit)):
            total += 1
            checksum, algorithm = _checksum(attachment)
            if not checksum:
                missing += 1
                continue
            checksummed += 1
            normalized_algorithm = _algorithm(algorithm)
            if normalized_algorithm:
                algorithms[normalized_algorithm] += 1
            checksum_groups[checksum].append(
                {
                    "checksum": checksum,
                    "unit_id": unit_id,
                    "attachment_id": _attachment_id(attachment, index),
                }
            )

    duplicate_groups = [
        {"checksum": checksum, "attachments": sorted(attachments, key=lambda item: (item["unit_id"], item["attachment_id"]))}
        for checksum, attachments in sorted(checksum_groups.items())
        if len(attachments) > 1
    ]

    return {
        "total_attachments": total,
        "checksummed_attachments": checksummed,
        "missing_checksum_count": missing,
        "duplicate_checksum_groups": duplicate_groups,
        "algorithms_used": [
            {"algorithm": algorithm, "count": count}
            for algorithm, count in sorted(algorithms.items(), key=lambda item: item[0])
        ],
        "largest_duplicate_group_size": max((len(group["attachments"]) for group in duplicate_groups), default=0),
    }


def _attachments(unit: Any) -> list[Mapping[str, Any]]:
    metadata = _metadata(unit)
    for key in _ATTACHMENT_KEYS:
        value = _get(unit, key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
        value = metadata.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
    return []


def _checksum(attachment: Mapping[str, Any]) -> tuple[str | None, str | None]:
    for key in _CHECKSUM_KEYS:
        value = attachment.get(key)
        if isinstance(value, str) and value.strip():
            algorithm = _first(attachment, _ALGORITHM_KEYS) or key
            return value.strip().lower(), str(algorithm)
    return None, None


def _algorithm(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().lower().replace("-", "")


def _attachment_id(attachment: Mapping[str, Any], index: int) -> str:
    value = attachment.get("id") or attachment.get("attachment_id") or attachment.get("filename") or index
    return str(value)


def _metadata(item: Any) -> Mapping[str, Any]:
    value = _get(item, "metadata")
    return value if isinstance(value, Mapping) else {}


def _unit_id(item: Any) -> str:
    for key in _UNIT_ID_KEYS:
        value = _get(item, key)
        if value not in (None, ""):
            return str(value)
    return ""


def _get(item: Any, key: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def _first(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None
