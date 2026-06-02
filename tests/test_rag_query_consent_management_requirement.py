from __future__ import annotations

import pytest

from graph.rag.query_consent_management_requirement import detect_query_consent_management_requirement


def test_detects_consent_categories_and_frameworks():
    result = detect_query_consent_management_requirement(
        "Need GDPR and IAB TCF evidence for cookie consent, opt-in, consent withdrawal, and consent records."
    )

    assert result == {
        "requires_consent_management": True,
        "cue_categories": ["opt_in", "consent_withdrawal", "cookie_consent", "consent_records"],
        "frameworks": ["GDPR", "IAB TCF"],
    }


def test_detects_opt_out_marketing_purpose_and_cmp_gpc():
    result = detect_query_consent_management_requirement(
        "How does the CMP honor GPC opt-out for marketing consent and consent by purpose?"
    )

    assert result["requires_consent_management"] is True
    assert result["cue_categories"] == ["opt_out", "marketing_consent", "purpose_consent"]
    assert result["frameworks"] == ["CMP", "GPC"]


def test_unrelated_privacy_query_without_consent_wording_returns_false():
    assert detect_query_consent_management_requirement("Summarize privacy policy data categories.") == {
        "requires_consent_management": False,
        "cue_categories": [],
        "frameworks": [],
    }


def test_empty_query_raises_value_error():
    with pytest.raises(ValueError):
        detect_query_consent_management_requirement("")
