from __future__ import annotations

from graph.rag.result_contradiction_hotspots import analyze_result_contradiction_hotspots


def test_result_contradiction_hotspots_groups_and_sorts_by_severity():
    rows = analyze_result_contradiction_hotspots(
        [
            {"metadata": {"entity": "Policy A"}, "snippet": "This contradicts the older memo."},
            {"topic": "Policy A", "content": "The finding is inconsistent and conflicts with trial data."},
            {"title": "Policy B", "text": "No issue."},
        ]
    )

    assert rows[0] == {"group_key": "policy-a", "result_count": 2, "contradiction_cue_count": 3, "matched_cues": ["conflicts", "contradicts", "inconsistent"], "severity": "high"}
    assert rows[1]["severity"] == "none"


def test_result_contradiction_hotspots_uses_source_fallback():
    rows = analyze_result_contradiction_hotspots([{"source": "Docs", "text": "Retracted claim."}])

    assert rows[0]["group_key"] == "docs"
    assert rows[0]["severity"] == "medium"
