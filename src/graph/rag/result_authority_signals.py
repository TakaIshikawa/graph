"""Authority signal scoring for retrieved results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlsplit

_SIGNALS = ("author", "publisher", "domain", "citations_count", "peer_reviewed", "official_source", "updated_at")


def analyze_result_authority_signals(results: Iterable[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        signals = {signal: _signal_value(result, signal) for signal in _SIGNALS}
        if not signals["domain"]:
            signals["domain"] = _domain(_first(result, ("url", "source_url", "link")))
        present = [key for key, value in signals.items() if _present(value)]
        missing = [key for key in _SIGNALS if key not in present]
        rows.append(
            {
                "result_id": _result_id(result, index),
                "authority_score": round(min(1.0, len(present) / len(_SIGNALS)), 2),
                "signals_present": present,
                "missing_signals": missing,
            }
        )
    return rows


def _signal_value(result: Any, key: str) -> Any:
    aliases = {"citations_count": ("citations_count", "citation_count", "citations")}
    return _first(result, aliases.get(key, (key,)))


def _first(item: Any, keys: tuple[str, ...]) -> Any:
    for container in (item, _value(item, "metadata"), _value(item, "unit"), _value(_value(item, "unit"), "metadata")):
        if container is None:
            continue
        for key in keys:
            value = _value(container, key)
            if value not in (None, ""):
                return value
    return None


def _value(item: Any, key: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def _present(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value > 0
    if isinstance(value, list | tuple | set | dict):
        return bool(value)
    return value not in (None, "")


def _domain(value: Any) -> str | None:
    if value in (None, ""):
        return None
    parsed = urlsplit(str(value) if "://" in str(value) else f"https://{value}")
    return parsed.hostname


def _result_id(result: Any, index: int) -> str:
    return str(_first(result, ("result_id", "id", "unit_id", "source_id")) or f"result-{index + 1}")
