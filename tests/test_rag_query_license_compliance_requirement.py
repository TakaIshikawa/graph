from __future__ import annotations

import pytest

from graph.rag.query_license_compliance_requirement import detect_query_license_compliance_requirement


def test_detects_license_compliance_cues_and_license_names():
    result = detect_query_license_compliance_requirement(
        "Review OSS license compliance, SPDX identifiers, copyleft obligations, MIT, Apache-2.0, and GPL."
    )

    assert result == {
        "requires_license_compliance": True,
        "cue_categories": ["license_compliance", "oss_license", "copyleft", "spdx"],
        "license_names": ["MIT", "Apache-2.0", "GPL"],
    }


def test_detects_commercial_attribution_redistribution_and_dependency_review():
    result = detect_query_license_compliance_requirement(
        "Need dependency license review for commercial use, attribution requirements, redistribution, BSD, MPL, and Creative Commons."
    )

    assert result["requires_license_compliance"] is True
    assert result["cue_categories"] == ["commercial_use", "attribution", "redistribution", "dependency_license_review"]
    assert result["license_names"] == ["BSD", "MPL", "Creative Commons"]


def test_generic_legal_query_without_license_wording_returns_false():
    assert detect_query_license_compliance_requirement("Summarize legal requirements for vendor onboarding.") == {
        "requires_license_compliance": False,
        "cue_categories": [],
        "license_names": [],
    }


def test_empty_query_raises_value_error():
    with pytest.raises(ValueError):
        detect_query_license_compliance_requirement("")
