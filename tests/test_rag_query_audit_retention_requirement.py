from graph.rag.query_audit_retention_requirement import detect_query_audit_retention_requirements


def test_detects_audit_log_retention_duration_and_immutability():
    result = detect_query_audit_retention_requirements("Audit-log retention must retain logs for 7 years in WORM storage.")

    assert result["has_audit_retention_requirements"] is True
    assert [row["category"] for row in result["requirements"]] == ["immutable_storage", "retention_duration"]


def test_detects_purge_archival_access_review_and_compliance():
    result = detect_query_audit_retention_requirements(
        "Audit logs need archive logs, log purge policy, access review evidence, and SOX retention."
    )

    assert [row["category"] for row in result["requirements"]] == [
        "access_review_evidence",
        "archival",
        "compliance_retention",
        "deletion_purge_policy",
    ]


def test_requires_audit_or_log_context_for_generic_retention():
    assert detect_query_audit_retention_requirements("Define data retention and archive records for customers.") == {
        "has_audit_retention_requirements": False,
        "requirements": [],
    }
