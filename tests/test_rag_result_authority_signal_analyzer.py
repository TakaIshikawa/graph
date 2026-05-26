from __future__ import annotations

from graph.rag import analyze_result_authority_signals


def test_authority_signal_scores_increase_for_explicit_metadata():
    rows = analyze_result_authority_signals(
        [
            {"id": "a", "url": "https://example.gov/report", "metadata": {"author": "Ada", "publisher": "Agency", "citations_count": 5, "peer_reviewed": True, "official_source": True, "updated_at": "2025-01-01"}},
            {"id": "b", "url": "https://blog.example/post"},
        ]
    )

    assert rows[0]["authority_score"] == 1.0
    assert rows[0]["signals_present"] == ["author", "publisher", "domain", "citations_count", "peer_reviewed", "official_source", "updated_at"]
    assert rows[1]["authority_score"] == 0.14
    assert "author" in rows[1]["missing_signals"]


def test_authority_signals_keep_input_order():
    assert [row["result_id"] for row in analyze_result_authority_signals([{"id": "b"}, {"id": "a"}])] == ["b", "a"]
