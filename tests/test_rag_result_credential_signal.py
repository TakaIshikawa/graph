from __future__ import annotations

from graph.rag.result_credential_signal import analyze_result_credential_signals


def test_result_credential_signal_reads_top_level_and_metadata_fields():
    summary = analyze_result_credential_signals(
        [
            {"id": "a", "title": "Paper", "author": "Jane Doe, PhD", "organization": "Example University"},
            {"id": "b", "metadata": {"affiliation": "Government standards body"}},
            {"id": "c", "byline": "Staff writer"},
        ]
    )

    assert summary["results_with_credentials"] == 2
    assert summary["results_without_credentials"] == 1
    assert summary["credential_signal_counts"]["phd"] == 1
    assert summary["credential_signal_counts"]["government"] == 1
    assert summary["organization_counts"] == {"Example University": 1, "Government standards body": 1}
