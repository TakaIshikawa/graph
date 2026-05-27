from __future__ import annotations

from graph.rag.evidence_sample_size_signal import extract_evidence_sample_size_signals


def test_parses_common_text_patterns():
    result = extract_evidence_sample_size_signals([{"id": "a", "text": "N = 1,200 with 300 respondents."}])

    assert result["records"][0]["sample_sizes"] == [1200, 300]


def test_includes_metadata_sample_size():
    result = extract_evidence_sample_size_signals([{"id": "a", "metadata": {"sample_size": "42"}}])

    assert result["records"][0]["signals"] == [{"source": "metadata", "value": 42}]


def test_aggregate_summary_reports_min_max_and_unknowns():
    result = extract_evidence_sample_size_signals([{"sample_size": 10}, {"text": "500 observations"}, {"text": "No sample."}])

    assert result["summary"]["total_records"] == 3
    assert result["summary"]["records_with_sample_size"] == 2
    assert result["summary"]["unknown_count"] == 1
    assert result["summary"]["min"] == 10
    assert result["summary"]["max"] == 500
