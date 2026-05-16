"""Scan retrieved RAG results for prompt-injection cues."""

from __future__ import annotations

import base64
import binascii
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

_MISSING = object()
_TEXT_KEYS = ("content", "text", "snippet")
_CUES: tuple[tuple[str, str, str, re.Pattern[str]], ...] = (
    ("ignore-instructions", "high", "ignore previous instructions", re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?\b", re.I)),
    ("system-prompt", "high", "system prompt", re.compile(r"\bsystem\s+(?:prompt|message|instructions?)\b", re.I)),
    ("developer-message", "high", "developer message", re.compile(r"\bdeveloper\s+(?:message|instructions?)\b", re.I)),
    ("reveal-secrets", "high", "reveal secrets", re.compile(r"\b(?:reveal|print|show|exfiltrate|leak)\s+(?:the\s+)?(?:secrets?|api[_ -]?keys?|tokens?|credentials?)\b", re.I)),
    ("tool-call-coercion", "medium", "tool call coercion", re.compile(r"\b(?:call|invoke|use|run)\s+(?:the\s+)?(?:tool|function|browser|shell|python)\b", re.I)),
    ("role-override", "medium", "role override", re.compile(r"\b(?:you are now|act as|new role:|forget your role)\b", re.I)),
)


def _payload(result: Any) -> Any:
    if isinstance(result, tuple) and result:
        return result[0]
    return result


def _field_value(item: Any, key: str) -> Any:
    if item is _MISSING or item is None:
        return _MISSING
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def _candidate_values(result: Any, key: str) -> Iterable[Any]:
    payload = _payload(result)
    value = _field_value(payload, key)
    if value is not _MISSING:
        yield value
    metadata = _field_value(payload, "metadata")
    if isinstance(metadata, Mapping):
        value = metadata.get(key, _MISSING)
        if value is not _MISSING:
            yield value
    unit = _field_value(payload, "unit")
    if unit is not _MISSING and unit is not None:
        value = _field_value(unit, key)
        if value is not _MISSING:
            yield value
        unit_metadata = _field_value(unit, "metadata")
        if isinstance(unit_metadata, Mapping):
            value = unit_metadata.get(key, _MISSING)
            if value is not _MISSING:
                yield value


def _first(result: Any, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        for value in _candidate_values(result, key):
            text = _string(value)
            if text is not None:
                return text
    return None


def _result_id(result: Any, index: int) -> str:
    return _first(result, ("id", "unit_id", "source_id")) or f"result-{index + 1}"


def _deobfuscate(text: str) -> str:
    compact = re.sub(r"[\W_]+", "", text).casefold()
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    decoded: list[str] = []
    for token in re.findall(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{16,}={0,2}(?![A-Za-z0-9+/=])", text):
        try:
            raw = base64.b64decode(token, validate=True)
        except (binascii.Error, ValueError):
            continue
        try:
            decoded.append(raw.decode("utf-8"))
        except UnicodeDecodeError:
            continue
    extras = [
        phrase
        for phrase in (
            "ignore previous instructions",
            "system prompt",
            "developer message",
            "reveal secrets",
        )
        if phrase.replace(" ", "") in compact
    ]
    return "\n".join([text, spaced, *decoded, *extras])


def _evidence(text: str, match: re.Match[str]) -> str:
    start = max(match.start() - 24, 0)
    end = min(match.end() + 24, len(text))
    return " ".join(text[start:end].split())[:120]


def scan_prompt_injection_risks(results: Iterable[Any]) -> dict[str, Any]:
    """Return prompt-injection risk rows for retrieved results."""
    risks: list[dict[str, Any]] = []
    cue_counts: Counter[str] = Counter()
    result_count = 0

    for index, result in enumerate(results):
        result_count += 1
        result_id = _result_id(result, index)
        content = "\n".join(
            text
            for key in _TEXT_KEYS
            for value in _candidate_values(result, key)
            if (text := _string(value)) is not None
        )
        if not content:
            continue
        searchable = _deobfuscate(content)
        matched_types: set[str] = set()
        for risk_type, severity, cue, pattern in _CUES:
            match = pattern.search(searchable)
            if match is None or risk_type in matched_types:
                continue
            matched_types.add(risk_type)
            cue_counts[risk_type] += 1
            risks.append(
                {
                    "result_id": result_id,
                    "type": risk_type,
                    "severity": severity,
                    "cue": cue,
                    "evidence": _evidence(searchable, match),
                }
            )

    severity_order = {"high": 0, "medium": 1, "low": 2}
    risks.sort(key=lambda row: (severity_order[row["severity"]], row["result_id"], row["type"]))
    return {
        "risk_count": len(risks),
        "risks": risks,
        "summary": {
            "result_count": result_count,
            "matched_cue_counts": dict(sorted(cue_counts.items())),
        },
    }
