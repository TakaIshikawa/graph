from __future__ import annotations

from graph.rag.evidence_access_barriers import analyze_evidence_access_barriers


def test_evidence_access_barriers_detects_common_signals():
    rows = analyze_evidence_access_barriers(
        [
            {
                "id": "paper",
                "title": "Trial report",
                "snippet": "Abstract only. Full text unavailable behind a subscription paywall.",
            },
            {"id": "archive", "metadata": {"note": "Archived copy available via Wayback."}},
        ]
    )

    assert rows[0]["barrier_labels"] == ["paywall", "abstract_only", "missing_full_text"]
    assert rows[0]["severity"] == "high"
    assert rows[1]["barrier_labels"] == ["archived_copy_available"]
    assert rows[1]["severity"] == "low"


def test_evidence_access_barriers_returns_none_for_open_item():
    rows = analyze_evidence_access_barriers([{"id": "open", "snippet": "Open access full text."}])

    assert rows[0]["barrier_labels"] == []
    assert rows[0]["severity"] == "none"
    assert rows[0]["mitigation_hints"] == []
