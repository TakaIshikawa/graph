from __future__ import annotations

from graph.rag.query_cold_start_requirement import detect_query_cold_start_requirement


def test_detects_serverless_cold_start_controls():
    result = detect_query_cold_start_requirement(
        "For serverless APIs, quantify cold start latency, use a warm pool, prewarming, "
        "provisioned concurrency, first-request penalty tracking, and scale-from-zero behavior."
    )

    assert result["has_cold_start_requirement"] is True
    assert result["categories"] == [
        "cold_start_latency",
        "warm_pool",
        "prewarming",
        "provisioned_concurrency",
        "first_request_penalty",
        "scale_from_zero",
    ]
    assert [row["category"] for row in result["requirements"]] == result["categories"]
    assert result["matched_phrases"] == [
        "cold start latency",
        "warm pool",
        "prewarming",
        "provisioned concurrency",
        "first-request penalty",
        "scale-from-zero",
    ]
    assert result["confidence"] == "high"


def test_detects_model_serving_min_instances_and_first_request_delay():
    result = detect_query_cold_start_requirement(
        "For model serving, compare min instances, warm-up requests, and initial request delay."
    )

    assert result["has_cold_start_requirement"] is True
    assert result["categories"] == ["provisioned_concurrency", "prewarming", "first_request_penalty"]
    assert result["confidence"] == "high"


def test_ordinary_startup_and_launch_wording_does_not_match():
    result = detect_query_cold_start_requirement("Summarize startup costs and the product launch plan.")

    assert result == {
        "has_cold_start_requirement": False,
        "requirements": [],
        "categories": [],
        "matched_phrases": [],
        "confidence": "none",
    }


def test_plain_launch_wording_without_cold_start_terms_does_not_match():
    result = detect_query_cold_start_requirement("What launch announcement should we prepare for the new feature?")

    assert result["has_cold_start_requirement"] is False
    assert result["categories"] == []

