from __future__ import annotations

from graph.store import summarize_unit_doi_hints


def test_doi_hint_summary_normalizes_sources_and_ignores_short_numbers():
    report = summarize_unit_doi_hints([
        {"id": "a", "content": "doi:10.1000/ABC.", "metadata": {"url": "https://doi.org/10.1000/abc"}},
        {"id": "b", "content": "version 10.2/abc is not a DOI", "metadata": {"note": "10.5555/XYZ"}},
    ])

    assert report["total_matches"] == 3
    assert report["units_with_matches"] == 2
    assert report["doi_values"][0]["doi"] == "10.1000/abc"
    assert report["doi_values"][0]["count"] == 2
