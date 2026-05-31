from __future__ import annotations

from graph.rag.result_paywall_signal import analyze_result_paywall_signals


def test_result_paywall_signals_report_metadata_snippets_and_open_access():
    report = analyze_result_paywall_signals(
        [
            {"id": "a", "paywall": True},
            {"id": "b", "snippet": "Subscription required for full text"},
            {"id": "c", "notes": "Login required. Abstract only."},
            {"id": "d", "open_access": True, "snippet": "Free full text"},
            {"id": "e"},
        ]
    )

    assert report["signal_counts"] == {"paywall": 1, "subscription": 1, "login_required": 1, "abstract_only": 1, "full_text_available": 2}
    assert report["result_ids_by_signal"]["full_text_available"] == ["b", "d"]
    assert report["affected_result_ids"] == ["a", "b", "c"]
