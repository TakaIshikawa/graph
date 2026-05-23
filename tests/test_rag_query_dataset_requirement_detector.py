from __future__ import annotations

import pytest

from graph.rag.query_dataset_requirement_detector import detect_query_dataset_requirements


FLAGS = (
    "requires_raw_data",
    "requires_tabular_data",
    "requires_benchmark_data",
    "requires_methodology_data",
    "requires_downloadable_source",
)


def test_detect_query_dataset_requirements_flags_direct_dataset_asks():
    payload = detect_query_dataset_requirements(
        "Find the downloadable CSV table, raw data, and benchmark dataset for this report."
    )

    assert list(payload) == [
        "normalized_query",
        *FLAGS,
        "matched_terms",
        "requirement_matches",
        "confidence",
    ]
    assert {flag: payload[flag] for flag in FLAGS} == {
        "requires_raw_data": True,
        "requires_tabular_data": True,
        "requires_benchmark_data": True,
        "requires_methodology_data": False,
        "requires_downloadable_source": True,
    }
    assert payload["matched_terms"] == ["raw data", "dataset", "table", "csv", "benchmark", "download"]
    assert payload["confidence"] == 0.99


def test_detect_query_dataset_requirements_flags_implicit_sample_size_asks():
    payload = detect_query_dataset_requirements(
        "How many participants were included, and is the methodology appendix available?"
    )

    assert payload["requires_methodology_data"] is True
    assert payload["requires_raw_data"] is False
    assert payload["requires_tabular_data"] is False
    assert payload["requires_benchmark_data"] is False
    assert payload["requires_downloadable_source"] is False
    assert payload["requirement_matches"]["requires_methodology_data"] == [
        "participants",
        "methodology",
        "appendix",
    ]
    assert payload["confidence"] == 0.53


def test_detect_query_dataset_requirements_leaves_unrelated_queries_unflagged():
    payload = detect_query_dataset_requirements("Explain why semantic search helps RAG answers.")

    assert {flag: payload[flag] for flag in FLAGS} == dict.fromkeys(FLAGS, False)
    assert payload["matched_terms"] == []
    assert payload["requirement_matches"] == {flag: [] for flag in FLAGS}
    assert payload["confidence"] == 0.0


@pytest.mark.parametrize("query", ["", "   ", None, 7])
def test_detect_query_dataset_requirements_validates_query(query):
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        detect_query_dataset_requirements(query)  # type: ignore[arg-type]
