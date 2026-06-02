from __future__ import annotations

from graph.rag.query_privacy_notice_requirement import detect_query_privacy_notice_requirement


def test_detects_privacy_notice_types():
    result = detect_query_privacy_notice_requirement(
        "Find privacy notice, notice at collection, privacy policy disclosure, and purpose disclosure language."
    )

    assert result == {
        "requires_privacy_notice": True,
        "cue_categories": ["privacy_notice", "privacy_policy_disclosure", "notice_at_collection", "purpose_disclosure"],
    }


def test_user_facing_notice_is_distinct_from_internal_policy():
    result = detect_query_privacy_notice_requirement("Do we have a user-facing data-use notice?")

    assert result["cue_categories"] == ["user_facing_notice"]


def test_generic_privacy_question_without_notice_or_disclosure_returns_false():
    assert detect_query_privacy_notice_requirement("Summarize privacy risks in the data flow.") == {
        "requires_privacy_notice": False,
        "cue_categories": [],
    }
