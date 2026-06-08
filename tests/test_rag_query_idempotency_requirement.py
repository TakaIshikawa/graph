from __future__ import annotations

from graph.rag.query_idempotency_requirement import detect_query_idempotency_requirement


def test_detects_idempotency_categories_and_severity():
    result = detect_query_idempotency_requirement(
        "For payment APIs, require idempotency keys, safe retries, duplicate request handling, "
        "and replayed submission rejection."
    )

    assert result["has_idempotency_requirement"] is True
    assert [row["category"] for row in result["requirements"]] == [
        "duplicate_request",
        "idempotency_key",
        "replayed_submission",
        "safe_retry",
    ]
    assert {row["category"]: row["severity"] for row in result["requirements"]} == {
        "duplicate_request": "high",
        "idempotency_key": "high",
        "replayed_submission": "medium",
        "safe_retry": "high",
    }


def test_returns_stable_sorted_rows_independent_of_phrase_order():
    result = detect_query_idempotency_requirement(
        "Explain idempotent retries, request deduplication, and Idempotency-Key behavior."
    )

    assert [row["category"] for row in result["requirements"]] == [
        "duplicate_request",
        "idempotency_key",
        "safe_retry",
    ]


def test_ignores_general_api_design_without_idempotency_cues():
    assert detect_query_idempotency_requirement("Compare general API design patterns for REST resources.") == {
        "has_idempotency_requirement": False,
        "requirements": [],
    }
