from __future__ import annotations

from types import SimpleNamespace

from graph.rag.source_authority_mix import analyze_source_authority_mix


def test_source_authority_mix_uses_explicit_metadata_before_domain_heuristics():
    report = analyze_source_authority_mix(
        [
            {"id": "gov", "source_type": "forum", "domain": "agency.gov"},
            {"id": "paper", "metadata": {"peer_reviewed": True, "domain": "blog.test"}},
            SimpleNamespace(id="docs", metadata={"verified": "yes", "publisher": "Vendor"}),
            {"id": "anon"},
        ]
    )

    assert report["tier_counts"] == {"high": 1, "medium": 1, "low": 1, "unknown": 1}
    assert report["tier_ratios"]["unknown"] == 0.25
    assert report["authority_score"] == 0.4625
    assert report["results"][0]["tier"] == "unknown"
    assert "anonymous_or_missing_authority" in report["warnings"]
    assert "low_authority_mix" in report["warnings"]
