"""Summarize source metadata completeness."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

DEFAULT_REQUIRED_KEYS = ("name", "url", "source_type")


def summarize_source_metadata_completeness(
    sources: Iterable[Mapping[str, Any] | object], required_keys: Sequence[str] | None = None, sample_limit: int = 5
) -> dict[str, Any]:
    keys = tuple(required_keys or DEFAULT_REQUIRED_KEYS)
    source_count = complete_source_count = 0
    missing_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []

    for index, source in enumerate(sources):
        source_count += 1
        metadata = _metadata(source)
        missing = [key for key in keys if not _present(_field(source, metadata, key))]
        if not missing:
            complete_source_count += 1
            continue
        missing_counts.update(missing)
        if len(samples) < sample_limit:
            samples.append({"source_id": _source_id(source, index), "missing_keys": missing})

    incomplete_source_count = source_count - complete_source_count
    return {
        "source_count": source_count,
        "required_keys": list(keys),
        "complete_source_count": complete_source_count,
        "incomplete_source_count": incomplete_source_count,
        "completeness_ratio": complete_source_count / source_count if source_count else 0,
        "missing_counts_by_key": [{"key": key, "count": missing_counts[key]} for key in sorted(missing_counts, key=lambda key: (-missing_counts[key], key.casefold(), key))],
        "samples": samples,
    }


def _metadata(source: Mapping[str, Any] | object) -> Mapping[str, Any]:
    value = source.get("metadata") if isinstance(source, Mapping) else getattr(source, "metadata", None)
    return value if isinstance(value, Mapping) else {}


def _field(source: Mapping[str, Any] | object, metadata: Mapping[str, Any], key: str) -> Any:
    value = source.get(key) if isinstance(source, Mapping) else getattr(source, key, None)
    return value if _present(value) else metadata.get(key)


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, Mapping | list | tuple | set):
        return bool(value)
    return True


def _source_id(source: Mapping[str, Any] | object, index: int) -> str:
    value = source.get("id") if isinstance(source, Mapping) else getattr(source, "id", None)
    text = "" if value is None else str(value).strip()
    return text or f"source:{index + 1}"
