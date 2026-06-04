from __future__ import annotations

from graph.rag import detect_query_right_to_erasure_requirement


def test_right_to_erasure_detects_deletion_backup_deadline_and_processors():
    result = detect_query_right_to_erasure_requirement(
        "For GDPR right to erasure, handle deletion requests within 30 days, "
        "delete from backups, propagate to processors, and keep audit records."
    )

    assert result["has_right_to_erasure_requirement"] is True
    assert [row["category"] for row in result["requirements"]] == [
        "audit_record",
        "backup_deletion",
        "deadline_sla",
        "deletion_request",
        "processor_propagation",
    ]


def test_right_to_erasure_detects_verification_and_retention_exceptions():
    result = detect_query_right_to_erasure_requirement(
        "Privacy deletion request flow needs identity verification and retention exceptions."
    )

    assert [row["category"] for row in result["requirements"]] == [
        "deletion_request",
        "identity_verification",
        "retention_exception",
    ]


def test_generic_delete_button_without_privacy_context_does_not_trigger():
    result = detect_query_right_to_erasure_requirement("Add a delete button for uploaded files and remove backups.")

    assert result["has_right_to_erasure_requirement"] is False
    assert result["requirements"] == []
