import pytest

from graph.rag.query_api_rate_quota_requirement import detect_query_api_rate_quota_requirement


def test_detects_api_quota_and_numeric_allowance():
    result = detect_query_api_rate_quota_requirement(
        "Need API quota, monthly call allowance of 100,000 calls per month, burst limit, and quota increase rules."
    )

    assert result["requires_api_rate_quota"] is True
    assert result["cue_categories"] == ["api_quota", "monthly_call_allowance", "burst_limit", "quota_increase"]
    assert result["numeric_quotas"] == ["100,000 calls per month"]


def test_detects_request_rate_limit_distinction():
    result = detect_query_api_rate_quota_requirement("Document request limits of 500 requests per day.")

    assert result["requires_api_rate_quota"] is True
    assert result["cue_categories"] == ["request_limit"]
    assert result["numeric_quotas"] == ["500 requests per day"]


def test_generic_api_integration_does_not_trigger_quota():
    assert detect_query_api_rate_quota_requirement("How do I integrate with the API endpoint?") == {
        "requires_api_rate_quota": False,
        "cue_categories": [],
        "numeric_quotas": [],
    }


@pytest.mark.parametrize("query", ["", None])
def test_invalid_query_raises_value_error(query):
    with pytest.raises(ValueError):
        detect_query_api_rate_quota_requirement(query)  # type: ignore[arg-type]
