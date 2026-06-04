from __future__ import annotations

from graph.rag.query_subprocessor_requirement import detect_query_subprocessor_requirements


def test_detects_subprocessor_and_sub_processor_list_disclosure():
    rows = detect_query_subprocessor_requirements("Require a subprocessor list and disclosure of sub-processors.")

    assert rows == [
        {"matched_text": "subprocessor list", "category": "list_disclosure", "severity": "high", "span": [10, 27]}
    ]


def test_detects_vendor_list_with_subprocessor_context():
    rows = detect_query_subprocessor_requirements("For subprocessors, include the vendor list in due diligence.")

    assert rows == [{"matched_text": "vendor list", "category": "list_disclosure", "severity": "high", "span": [31, 42]}]


def test_detects_change_notifications_and_objection_rights():
    rows = detect_query_subprocessor_requirements(
        "Subprocessor change notifications must include objection rights and 30 days to object."
    )

    assert rows == [
        {"matched_text": "change notifications", "category": "notification", "severity": "high", "span": [13, 33]},
        {"matched_text": "objection rights", "category": "objection_rights", "severity": "high", "span": [47, 63]},
    ]


def test_detects_onward_transfer_location_and_dpia_support():
    rows = detect_query_subprocessor_requirements(
        "DPA review should cover subprocessor locations, onward transfers, and DPIA support."
    )

    assert rows == [
        {"matched_text": "subprocessor locations", "category": "data_location", "severity": "medium", "span": [24, 46]},
        {"matched_text": "DPIA support", "category": "dpia_support", "severity": "medium", "span": [70, 82]},
        {"matched_text": "onward transfers", "category": "onward_transfer", "severity": "high", "span": [48, 64]},
    ]


def test_generic_vendor_wording_without_privacy_or_subprocessor_context_does_not_match():
    assert detect_query_subprocessor_requirements("Compare vendor list quality and change notifications.") == []


def test_span_positions_point_to_matched_text():
    query = "Need privacy evidence for the right to object."
    rows = detect_query_subprocessor_requirements(query)

    assert rows == [
        {"matched_text": "right to object", "category": "objection_rights", "severity": "high", "span": [30, 45]}
    ]
    start, end = rows[0]["span"]
    assert query[start:end] == rows[0]["matched_text"]


def test_categories_are_sorted_by_category_name():
    rows = detect_query_subprocessor_requirements(
        "Subprocessors need onward transfers, vendor list, DPIA support, objection rights, data location, and change notification."
    )

    assert [row["category"] for row in rows] == sorted(row["category"] for row in rows)
