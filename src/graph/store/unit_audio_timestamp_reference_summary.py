"""Summarize audio file references with nearby timestamp cues."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, unit_id

_AUDIO_KEYS = ("audio_url", "media_url", "source_url", "path")
_EXTENSIONS = ("mp3", "m4a", "wav", "flac")
_AUDIO_RE = re.compile(r"(?P<ref>[^\s<>()\"']+?\.(?P<ext>mp3|m4a|wav|flac)(?:[?#][^\s<>()\"']*)?)", re.IGNORECASE)
_TIMESTAMP_RE = re.compile(r"(?<!\d)(?:\d{1,2}:)?[0-5]?\d:[0-5]\d(?!\d)")
_NEARBY_CHARS = 80


def summarize_unit_audio_timestamp_references(units: Iterable[Any], *, sample_limit: int = 5) -> dict[str, Any]:
    """Return deterministic counts for audio references and nearby timestamps."""

    extension_counts: Counter[str] = Counter({extension: 0 for extension in _EXTENSIONS})
    timestamped = untimestamped = total = 0
    samples: list[dict[str, Any]] = []

    for unit in units:
        uid = unit_id(unit)
        for reference, extension, has_timestamp in _references(unit):
            total += 1
            extension_counts[extension] += 1
            if has_timestamp:
                timestamped += 1
            else:
                untimestamped += 1
            if len(samples) < max(0, sample_limit):
                samples.append(
                    {
                        "unit_id": uid,
                        "reference": reference,
                        "extension": extension,
                        "timestamped": has_timestamp,
                    }
                )

    samples.sort(key=lambda row: (sort_key(row["unit_id"]), sort_key(row["reference"])))
    return {
        "total_references": total,
        "extension_counts": dict(sorted(extension_counts.items(), key=lambda item: sort_key(item[0]))),
        "timestamped_reference_count": timestamped,
        "untimestamped_reference_count": untimestamped,
        "samples": samples[: max(0, sample_limit)],
    }


def _references(unit: Any) -> list[tuple[str, str, bool]]:
    rows: list[tuple[str, str, bool]] = []
    for text in _texts(unit):
        for match in _AUDIO_RE.finditer(text):
            reference = match.group("ref").rstrip(".,;:]}")
            extension = match.group("ext").casefold()
            window = text[max(0, match.start() - _NEARBY_CHARS) : min(len(text), match.end() + _NEARBY_CHARS)]
            rows.append((reference, extension, bool(_TIMESTAMP_RE.search(window))))
    return rows


def _texts(unit: Any) -> list[str]:
    meta = metadata(unit)
    values = [get(unit, "content")]
    for key in _AUDIO_KEYS:
        values.append(get(unit, key))
        values.append(meta.get(key))
    return [text for value in values for text in _flatten_text(value)]


def _flatten_text(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, Mapping):
        return [item for child in value.values() for item in _flatten_text(child)]
    if isinstance(value, list | tuple | set):
        return [item for child in value for item in _flatten_text(child)]
    text = field_value(value)
    return [text] if text else []
