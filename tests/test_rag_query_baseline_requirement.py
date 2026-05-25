from __future__ import annotations

from graph.rag.query_baseline_requirement import detect_query_baseline_requirement


def test_detects_baseline_before_after_and_previous_period_cues():
    report = detect_query_baseline_requirement("Compare adoption compared with last year, before vs after launch, against current state.")

    assert report["requires_baseline"] is True
    assert [row["type"] for row in report["matched_cues"]] == ["previous_period", "before_after", "baseline"]
    assert report["baseline_anchors"] == ["current state", "last year"]


def test_extracts_prelaunch_anchor():
    report = detect_query_baseline_requirement("Show conversion relative to baseline and pre-launch.")

    assert report["requires_baseline"] is True
    assert "baseline" in report["baseline_anchors"]
    assert report["matched_cues"][1]["type"] == "before_after"


def test_generic_comparison_without_baseline_context_is_neutral():
    report = detect_query_baseline_requirement("Which option is larger than the other?")

    assert report == {"requires_baseline": False, "matched_cues": [], "baseline_anchors": []}
