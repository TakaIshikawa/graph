from graph.rag.query_break_glass_requirement import detect_query_break_glass_requirements


def test_break_glass_and_emergency_access_wording_are_detected():
    report = detect_query_break_glass_requirements("Define break-glass access and emergency access.")

    assert report == {
        "has_break_glass_requirements": True,
        "requirements": ["break_glass_access", "emergency_access"],
        "post_use_review_required": False,
    }


def test_approval_monitoring_and_post_use_review_are_normalized_separately():
    report = detect_query_break_glass_requirements(
        "Require emergency admin, privileged emergency account approval for emergency access, monitoring emergency access, and review after use."
    )

    assert report["requirements"] == [
        "emergency_access",
        "emergency_admin",
        "privileged_emergency_account",
        "approval",
        "monitoring",
        "post_use_review",
    ]
    assert report["post_use_review_required"] is True


def test_unrelated_query_has_no_break_glass_requirements():
    assert detect_query_break_glass_requirements("Review ordinary administrator login metrics.") == {
        "has_break_glass_requirements": False,
        "requirements": [],
        "post_use_review_required": False,
    }
