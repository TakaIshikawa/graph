from __future__ import annotations

from graph.rag.evidence_statistical_claim_audit import audit_evidence_statistical_claims


def test_evidence_statistical_claim_audit_empty_input():
    assert audit_evidence_statistical_claims([]) == {"findings": []}


def test_evidence_statistical_claim_audit_detects_percentage_with_missing_context():
    result = audit_evidence_statistical_claims([{"id": "e1", "text": "Completion increased by 42%."}])

    finding = result["findings"][0]
    assert finding["evidence_id"] == "e1"
    assert finding["statistic_type"] == "percentage"
    assert set(finding["missing_context"]) >= {"denominator", "timeframe", "population"}
    assert finding["support_status"] == "missing_context"


def test_evidence_statistical_claim_audit_classifies_supported_ratio_average_and_rate():
    result = audit_evidence_statistical_claims(
        [
            {"id": "r", "text": "In 2024, 8 out of 10 users in the sample completed the task."},
            {"id": "a", "text": "During 2024, among users, the average time was 4 hours per task."},
            {"id": "p", "text": "In 2024, among patients, the rate was 5 cases per week."},
        ]
    )

    assert [row["statistic_type"] for row in result["findings"]] == ["ratio", "average", "rate"]
    assert result["findings"][0]["support_status"] == "sufficient"
