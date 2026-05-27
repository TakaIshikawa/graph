from __future__ import annotations

from graph.rag.query_metric_requirement import detect_query_metric_requirement


def test_detects_kpi_and_threshold_queries():
    result = detect_query_metric_requirement("Compare KPIs against a 95% accuracy threshold and latency benchmarks.")

    assert result["required"] is True
    assert result["metric_terms"] == ["kpi", "benchmark", "threshold", "percentage", "latency", "accuracy"]
    assert result["numeric_cues"] == ["95%"]
    assert "threshold" in result["suggested_evidence_fields"]


def test_deduplicates_metric_terms():
    result = detect_query_metric_requirement("Show cost metrics, costs, ROI and return on investment.")

    assert result["metric_terms"] == ["cost", "roi", "metric"]


def test_non_metric_how_to_query_is_not_required():
    result = detect_query_metric_requirement("How do I configure the importer for markdown notes?")

    assert result["required"] is False
    assert result["metric_terms"] == []
    assert result["numeric_cues"] == []
