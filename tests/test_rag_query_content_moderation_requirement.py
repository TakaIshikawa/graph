from graph.rag.query_content_moderation_requirement import detect_query_content_moderation_requirement


def test_moderation_and_ugc_safety_trigger():
    report = detect_query_content_moderation_requirement("Add content moderation for user-generated content safety.")

    assert report == {
        "requires_content_moderation": True,
        "policy_areas": ["moderation", "ugc_safety"],
        "human_review_required": False,
        "matched_cues": ["moderation", "ugc_safety"],
        "severity": "medium",
    }


def test_policy_areas_and_human_review_are_extracted():
    report = detect_query_content_moderation_requirement(
        "Detect spam, hate speech, harassment, and CSAM with moderator review."
    )

    assert report["policy_areas"] == ["hate", "harassment", "spam", "csam"]
    assert report["human_review_required"] is True
    assert report["severity"] == "high"


def test_unrelated_query_has_no_moderation_requirement():
    assert detect_query_content_moderation_requirement("Summarize content marketing examples.") == {
        "requires_content_moderation": False,
        "policy_areas": [],
        "human_review_required": False,
        "matched_cues": [],
        "severity": "none",
    }
