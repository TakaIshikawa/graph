from graph.rag.query_policy_exception_requirement import detect_query_policy_exception_requirement


def test_policy_exception_signals_are_detected():
    report = detect_query_policy_exception_requirement(
        "Need a policy exception, waiver, exception approval, risk acceptance, and temporary exemption."
    )

    assert report["requires_policy_exception"] is True
    assert report["categories"] == [
        "policy_exception",
        "waiver",
        "exception_approval",
        "risk_acceptance",
        "temporary_exemption",
    ]
    assert report["matches"][0]["span"] == (7, 23)


def test_compensating_control_and_risk_acceptance_are_high_severity():
    report = detect_query_policy_exception_requirement("Use compensating controls with risk acceptance.")

    assert report["categories"] == ["compensating_control", "risk_acceptance"]
    assert [match["severity"] for match in report["matches"]] == ["high", "high"]


def test_unrelated_policy_query_returns_empty_result():
    assert detect_query_policy_exception_requirement("Find the current password policy.") == {
        "requires_policy_exception": False,
        "categories": [],
        "matches": [],
    }
