from __future__ import annotations

from graph.rag.context_authority_gaps import analyze_context_authority_gaps


def test_reports_present_and_missing_authority_tiers():
    report = analyze_context_authority_gaps(
        [{"id": "gov", "metadata": {"authority_tier": "primary"}}, {"id": "pub", "url": "https://example.com/a"}],
        required_tiers=["primary", "expert"],
    )

    assert report["present_tiers"] == ["primary", "publisher"]
    assert report["missing_tiers"] == ["expert"]
    assert report["tier_counts"]["primary"] == 1
    assert "missing_required_authority_tiers" in report["warnings"]


def test_explicit_metadata_overrides_domain_heuristics_and_query_defaults():
    report = analyze_context_authority_gaps(
        [{"id": "forum", "source_type": "community", "url": "https://agency.gov/doc"}],
        query="medical dose guidance",
    )

    assert report["required_tiers"] == ["primary", "expert"]
    assert report["results"][0]["authority_tier"] == "community"
    assert report["results"][0]["reason"] == "explicit_metadata"
