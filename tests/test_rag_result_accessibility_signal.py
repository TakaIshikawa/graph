from __future__ import annotations

from types import SimpleNamespace

from graph.rag.result_accessibility_signal import analyze_result_accessibility_signals


def test_result_accessibility_signal_classifies_records():
    summary = analyze_result_accessibility_signals(
        [
            {"id": "open", "open_access": True},
            {"id": "login", "metadata": {"login_required": True}},
            {"id": "media", "pdf_url": "https://x.test/file.pdf", "transcript_url": "https://x.test/t"},
            SimpleNamespace(id="unknown"),
        ]
    )

    assert summary["accessible_count"] == 2
    assert summary["restricted_count"] == 1
    assert summary["unknown_count"] == 1
    assert summary["signal_counts"]["pdf_url"] == 1
    assert summary["samples"][1]["status"] == "restricted"
