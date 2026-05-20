from __future__ import annotations

from graph.rag.query_revision_plan import build_query_revision_plan


def test_query_revision_plan_handles_empty_analysis_with_conservative_suggestion():
    report = build_query_revision_plan("hybrid search", {})

    assert report["suggestions"][0]["code"] == "conservative_broadening"
    assert report["suggestions"][0]["priority"] == 5


def test_query_revision_plan_maps_multiple_warnings_and_dedupes():
    report = build_query_revision_plan(
        "hybrid search",
        {"warnings": ["stale_evidence", "too_few_sources", "too_few_sources", "uncited_claims"]},
    )

    codes = [item["code"] for item in report["suggestions"]]
    assert codes == ["citation_support", "refresh_time_filter", "broaden_sources"]
    assert len(codes) == len(set(codes))
    assert report["suggestions"][0]["revised_query"] == "hybrid search citable sources"


def test_query_revision_plan_reports_unknown_warning_codes():
    report = build_query_revision_plan("hybrid search", {"warnings": ["unknown_warning"]})

    assert report["suggestions"] == []
    assert report["ignored_warnings"] == ["unknown_warning"]
