from __future__ import annotations

from graph.rag.query_verification_depth_requirement import detect_query_verification_depth_requirement


def test_query_verification_depth_detects_deep_cues():
    report = detect_query_verification_depth_requirement("Audit this and source every claim.")

    assert report["required_depth"] == "deep"
    assert report["suggested_retrieval_passes"] == 3
    assert [cue["cue"] for cue in report["matched_cues"]] == ["audit", "source every claim"]


def test_query_verification_depth_quick_check_stays_shallow():
    report = detect_query_verification_depth_requirement("Quick check whether the summary matches.")

    assert report["required_depth"] == "shallow"
    assert report["confidence"] == 0.75
    assert report["suggested_retrieval_passes"] == 1


def test_query_verification_depth_deep_cues_take_precedence_and_no_cue_falls_back():
    mixed = detect_query_verification_depth_requirement("Sanity check and verify the figures.")
    fallback = detect_query_verification_depth_requirement("Summarize the figures.")

    assert mixed["required_depth"] == "deep"
    assert fallback["required_depth"] == "normal"
    assert fallback["matched_cues"] == []
