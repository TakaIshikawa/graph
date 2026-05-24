from __future__ import annotations

import pytest

from graph.rag.query_source_licensing_requirement import detect_query_source_licensing_requirement


def test_source_licensing_detects_named_licenses_and_reuse_constraints():
    result = detect_query_source_licensing_requirement(
        "Find Creative Commons CC-BY or public domain sources for commercial use with attribution required."
    )

    assert result["requires_license_filtering"] is True
    assert result["license_cues"] == ["creative_commons", "public_domain"]
    assert result["reuse_cues"] == ["commercial_use", "attribution_required"]
    assert result["restricted_use_cues"] == []
    assert result["recommendations"] == [
        "filter_sources_by_machine_readable_license_metadata",
        "preserve_attribution_and_reuse_terms_in_citations",
    ]


def test_source_licensing_separates_restricted_use_cues():
    result = detect_query_source_licensing_requirement("Avoid proprietary or non-commercial fair use material.")

    assert result["license_cues"] == ["fair_use", "proprietary"]
    assert result["restricted_use_cues"] == ["proprietary", "noncommercial_only", "fair_use_only"]


def test_source_licensing_no_cues_is_false():
    result = detect_query_source_licensing_requirement("Find primary sources.")

    assert result["requires_license_filtering"] is False
    assert result["confidence"] == 0.0


@pytest.mark.parametrize("query", ["", None])
def test_source_licensing_validates_query(query):
    with pytest.raises(ValueError):
        detect_query_source_licensing_requirement(query)  # type: ignore[arg-type]
