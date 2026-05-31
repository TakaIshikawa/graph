from __future__ import annotations

from types import SimpleNamespace

from graph.rag.result_source_type_coverage import analyze_result_source_type_coverage


def test_result_source_type_coverage_classifies_inputs_and_expected_gaps():
    report = analyze_result_source_type_coverage(
        [
            {"id": "docs", "metadata": {"source_type": "documentation"}, "title": "API Guide"},
            SimpleNamespace(id="paper", url="https://arxiv.org/abs/1", title="Study"),
            {"id": "repo", "url": "https://github.com/org/repo"},
            {"id": "mystery", "title": "Untyped"},
        ],
        expected_types=["docs", "academic", "news"],
    )

    assert report["type_counts"]["docs"] == 1
    assert report["type_counts"]["academic"] == 1
    assert report["type_counts"]["code"] == 1
    assert report["type_counts"]["unknown"] == 1
    assert report["covered_types"] == ["docs", "academic", "code"]
    assert report["missing_expected_types"] == ["news"]
    assert report["examples"]["docs"] == [{"id": "docs", "title": "API Guide"}]
