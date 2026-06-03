from __future__ import annotations

from graph.rag.query_rate_limit_backoff_requirement import detect_query_rate_limit_backoff_requirement


def test_detects_429_exponential_backoff_retry_after_and_jitter_guidance():
    result = detect_query_rate_limit_backoff_requirement(
        "For the API client, handle HTTP 429 with exponential backoff and Retry-After support."
    )

    assert result["requires_rate_limit_backoff"] is True
    assert result["backoff_terms"] == ["exponential_backoff"]
    assert result["retry_terms"] == ["http_429", "retry_after"]
    assert result["matched_phrases"] == ["HTTP 429", "exponential backoff", "Retry-After"]
    assert result["confidence"] == "high"
    assert any("Retry-After" in recommendation for recommendation in result["recommendations"])
    assert any("jitter" in recommendation for recommendation in result["recommendations"])


def test_detects_jitter_idempotent_retries_and_throttling_recovery():
    result = detect_query_rate_limit_backoff_requirement(
        "Need throttling recovery docs with jitter and idempotent retries for failed requests."
    )

    assert result["requires_rate_limit_backoff"] is True
    assert result["backoff_terms"] == ["throttling_recovery", "jitter"]
    assert result["retry_terms"] == ["idempotent_retries"]
    assert result["confidence"] == "high"


def test_plain_quota_or_pricing_limit_question_does_not_trigger_backoff():
    result = detect_query_rate_limit_backoff_requirement(
        "Compare API quota, monthly request limits, free tier pricing, and paid overage limits."
    )

    assert result == {
        "requires_rate_limit_backoff": False,
        "backoff_terms": [],
        "retry_terms": [],
        "matched_phrases": [],
        "recommendations": [],
        "confidence": "none",
    }


def test_retry_policy_without_rate_limit_context_stays_low_confidence_negative():
    result = detect_query_rate_limit_backoff_requirement("Should students retry the policy quiz after failing?")

    assert result["requires_rate_limit_backoff"] is False
    assert result["retry_terms"] == []
    assert result["recommendations"] == []
    assert result["confidence"] == "none"
