from __future__ import annotations

from graph.rag.query_records_management_requirement import detect_query_records_management_requirement


def test_detects_records_management_lifecycle_signals():
    result = detect_query_records_management_requirement(
        "Need records management, recordkeeping, records schedule, official record, archive policy, and records disposition evidence."
    )

    assert result == {
        "requires_records_management": True,
        "cue_categories": [
            "records_management",
            "recordkeeping",
            "records_schedule",
            "disposition",
            "archive_policy",
            "official_record",
        ],
    }


def test_retention_adjacent_wording_requires_records_management_language():
    assert detect_query_records_management_requirement("Compare retention periods for logs.") == {
        "requires_records_management": False,
        "cue_categories": [],
    }

    assert detect_query_records_management_requirement("Compare the records schedule for retained files.")[
        "requires_records_management"
    ] is True


def test_unrelated_archive_or_history_query_does_not_match():
    assert detect_query_records_management_requirement("Search the product history archive for release notes.") == {
        "requires_records_management": False,
        "cue_categories": [],
    }
