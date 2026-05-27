from __future__ import annotations

from graph.rag.query_version_requirement import detect_query_version_requirement


def test_detect_query_version_requirement_extracts_versions_and_latest_lts():
    report = detect_query_version_requirement(
        "Use Python 3.12 with the v2 API and latest LTS Node release."
    )

    assert report == {
        "has_version_requirement": True,
        "versions": ["3.12", "2"],
        "compatibility_cues": [],
        "freshness_sensitive": True,
    }


def test_detect_query_version_requirement_reports_years_and_compatibility_separately():
    report = detect_query_version_requirement(
        "Find backward compatible guidance for the 2024 edition that works with version 1."
    )

    assert report == {
        "has_version_requirement": True,
        "versions": ["1", "2024"],
        "compatibility_cues": ["backward_compatible", "compatibility_target"],
        "freshness_sensitive": False,
    }


def test_detect_query_version_requirement_handles_queries_without_version_language():
    report = detect_query_version_requirement("How should I structure the answer?")

    assert report == {
        "has_version_requirement": False,
        "versions": [],
        "compatibility_cues": [],
        "freshness_sensitive": False,
    }
