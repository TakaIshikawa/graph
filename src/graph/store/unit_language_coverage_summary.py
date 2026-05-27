"""Language coverage summary for store units."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

LANGUAGE_KEYS = ("language", "lang", "locale", "detected_language")


def summarize_unit_language_coverage(units: Iterable[Any]) -> dict[str, Any]:
    total = with_language = 0
    counts: Counter[str] = Counter()
    for unit in units:
        total += 1
        language = _language(unit)
        if language:
            with_language += 1
            counts[language] += 1
    return {
        "total_units": total,
        "units_with_language": with_language,
        "units_without_language": total - with_language,
        "coverage_ratio": f"{(with_language / total):.2f}" if total else "0.00",
        "language_counts": {key: counts[key] for key in sorted(counts, key=_sort_key)},
    }


def _language(unit: Any) -> str:
    meta = _metadata(unit)
    for key in LANGUAGE_KEYS:
        text = _text(_get(unit, key)) or _text(meta.get(key))
        if text:
            return text.replace("_", "-").split("-")[0].casefold()
    return ""


def _metadata(unit: Any) -> Mapping[str, Any]:
    value = _get(unit, "metadata")
    return value if isinstance(value, Mapping) else {}


def _get(item: Any, key: str) -> Any:
    return item.get(key) if isinstance(item, Mapping) else getattr(item, key, None)


def _text(value: Any) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _sort_key(value: Any) -> tuple[str, str]:
    text = _text(value)
    return (text.casefold(), text)
