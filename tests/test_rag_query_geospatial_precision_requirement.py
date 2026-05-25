from __future__ import annotations

import pytest

from graph.rag.query_geospatial_precision_requirement import detect_query_geospatial_precision_requirement


def test_geospatial_precision_detects_levels_coordinates_and_radius():
    result = detect_query_geospatial_precision_requirement(
        "Find city-level results with exact address and latitude within 5 km."
    )

    assert result["requires_geospatial_precision"] is True
    assert result["precision_levels"] == ["city", "address", "coordinates"]
    assert result["distance_constraints"] == [{"value": "5", "unit": "kilometers"}]
    assert result["coordinate_cues"] == ["latitude"]
    assert result["recommendations"] == [
        "retrieve_sources_with_matching_location_granularity",
        "apply_radius_filter_before_ranking_results",
        "preserve_coordinate_precision_and_coordinate_reference_system",
    ]


def test_geospatial_precision_ranks_broad_to_exact():
    result = detect_query_geospatial_precision_requirement("Compare country-level, county, ZIP code, and coordinates.")

    assert result["precision_levels"] == ["country", "county", "postal_code", "coordinates"]


def test_geospatial_precision_no_cues_is_false():
    result = detect_query_geospatial_precision_requirement("Explain retrieval quality.")

    assert result["requires_geospatial_precision"] is False
    assert result["confidence"] == 0.0


@pytest.mark.parametrize("query", ["", " ", None])
def test_geospatial_precision_validates_query(query):
    with pytest.raises(ValueError):
        detect_query_geospatial_precision_requirement(query)  # type: ignore[arg-type]
