from __future__ import annotations

from graph.rag.result_geographic_coverage import analyze_result_geographic_coverage


def test_extracts_locations_from_metadata_and_text():
    result = analyze_result_geographic_coverage(
        "Compare adoption.",
        [
            {"id": "r1", "metadata": {"country": "United States"}},
            {"id": "r2", "title": "Energy policy in Europe"},
            {"id": "r3", "content": "A global overview"},
        ],
    )

    assert result["location_counts"]["United States"] == 1
    assert result["location_counts"]["Europe"] == 1
    assert result["coverage_type_counts"]["global"] == 1


def test_flags_single_region_concentration_for_global_query():
    result = analyze_result_geographic_coverage(
        "Need global cross-country evidence.",
        [
            {"id": "r1", "country": "United States"},
            {"id": "r2", "country": "United States"},
            {"id": "r3", "country": "United States"},
            {"id": "r4", "country": "Canada"},
        ],
    )

    assert "single_region_concentration" in result["concentration_warnings"]


def test_reports_missing_requested_location():
    result = analyze_result_geographic_coverage("Evidence for Japan", [{"id": "r1", "country": "Canada"}])

    assert result["missing_location_hints"] == ["Japan"]


def test_empty_results_are_stable():
    result = analyze_result_geographic_coverage("global evidence", [])

    assert result["result_count"] == 0
    assert result["location_counts"] == {}
    assert result["concentration_warnings"] == ["no_results"]
