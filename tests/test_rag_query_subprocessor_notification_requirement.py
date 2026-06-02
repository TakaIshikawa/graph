from graph.rag.query_subprocessor_notification_requirement import (
    detect_query_subprocessor_notification_requirements,
)


def test_detects_subprocessor_list_and_notification_requirements():
    rows = detect_query_subprocessor_notification_requirements("Require a subprocessor list and subprocessor notification.")

    assert rows == [
        {"matched_text": "subprocessor list", "category": "subprocessor_list", "requirement_strength": "high"},
        {"matched_text": "subprocessor notification", "category": "subprocessor_notice", "requirement_strength": "high"},
    ]


def test_detects_vendor_change_notice_objection_and_processor_updates():
    rows = detect_query_subprocessor_notification_requirements(
        "Need vendor change notice, 30 days to object, and third-party processor updates."
    )

    assert rows == [
        {"matched_text": "30 days to object", "category": "objection_period", "requirement_strength": "high"},
        {
            "matched_text": "third-party processor updates",
            "category": "third_party_processor_update",
            "requirement_strength": "medium",
        },
        {"matched_text": "vendor change notice", "category": "vendor_change_notice", "requirement_strength": "high"},
    ]


def test_unrelated_third_party_integration_query_does_not_match():
    assert detect_query_subprocessor_notification_requirements("Compare third-party integration setup steps.") == []
