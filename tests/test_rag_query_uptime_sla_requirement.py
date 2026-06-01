from graph.rag import detect_query_uptime_sla_requirement


def test_uptime_and_availability_commitments_are_detected():
    report = detect_query_uptime_sla_requirement("What is the uptime SLA and service availability target?")

    assert report["requires_uptime_sla"] is True
    assert "availability_target" in report["cue_categories"]


def test_percentage_targets_are_extracted():
    report = detect_query_uptime_sla_requirement("Do you commit to 99.9% monthly uptime percentage?")

    assert report["percentage_targets"] == ["99.9%"]
    assert report["requires_uptime_sla"] is True


def test_sla_credit_only_question_does_not_match():
    assert detect_query_uptime_sla_requirement("What service credits are paid for an SLA breach?")["requires_uptime_sla"] is False
