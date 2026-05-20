from __future__ import annotations

from graph.rag.query_source_coverage import plan_query_source_coverage


def test_maps_query_cues_to_required_source_categories():
    report = plan_query_source_coverage(
        "Compare the history of a tax regulation and troubleshoot an error",
        [
            {"id": "official", "source_type": "government"},
            {"id": "forum", "metadata": {"source_category": "community"}},
        ],
    )

    assert report["matched_cues"] == ["comparison", "timeline", "legal", "financial", "troubleshooting"]
    assert "comparative" in report["required_categories"]
    assert "timeline" in report["required_categories"]
    assert "troubleshooting" in report["required_categories"]
    assert "expert" in report["missing_categories"]


def test_metadata_source_category_precedes_domain_heuristic():
    report = plan_query_source_coverage(
        "debug error",
        [{"id": "doc", "source_category": "troubleshooting", "url": "https://example.edu/help"}],
    )

    assert report["results"][0]["source_category"] == "troubleshooting"
    assert report["results"][0]["reason"] == "explicit_metadata"
