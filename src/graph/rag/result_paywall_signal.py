"""Analyze paywall and access-limitation signals in RAG results."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, result_id, string, value

_SIGNALS = {
    "paywall": ("paywall", "paid access", "purchase required"),
    "subscription": ("subscription", "subscriber only", "subscribe to"),
    "login_required": ("login required", "log in", "sign in", "account required"),
    "abstract_only": ("abstract only", "abstract available", "no full text"),
    "full_text_available": ("full text", "open access", "free full text"),
}


def analyze_result_paywall_signals(results: Iterable[Any]) -> dict[str, Any]:
    rows = list(results or [])
    ids_by_signal: dict[str, list[str]] = defaultdict(list)
    for index, result in enumerate(rows):
        rid = result_id(result, index)
        for signal in _detect(result):
            ids_by_signal[signal].append(rid)
    counts = {signal: len(ids_by_signal.get(signal, [])) for signal in _SIGNALS}
    affected = sorted({rid for signal, ids in ids_by_signal.items() if signal != "full_text_available" for rid in ids})
    return {"total_results": len(rows), "signal_counts": counts, "result_ids_by_signal": {signal: ids_by_signal.get(signal, []) for signal in _SIGNALS}, "affected_result_ids": affected}


def _detect(result: Any) -> set[str]:
    found = set()
    bool_map = {"paywall": "paywall", "is_paywalled": "paywall", "subscription_required": "subscription", "login_required": "login_required", "full_text_available": "full_text_available", "open_access": "full_text_available", "is_open_access": "full_text_available"}
    for key, signal in bool_map.items():
        if value(result, key) is True:
            found.add(signal)
    text = " ".join(part for part in [string(value(result, "access")) or "", string(value(result, "access_status")) or "", string(value(result, "notes")) or "", content_text(result)] if part).casefold()
    for signal, cues in _SIGNALS.items():
        if any(cue in text for cue in cues):
            found.add(signal)
    return found
