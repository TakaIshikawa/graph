from __future__ import annotations

from graph.rag.query_comparison_operator_detector import detect_query_comparison_operators


def test_query_comparison_operator_detector_empty_query():
    assert detect_query_comparison_operators("") == []


def test_query_comparison_operator_detector_normalizes_multiple_classes():
    rows = detect_query_comparison_operators("Compare top 5 funds vs bonds after 2020 with fees less than 1%.")

    kinds = [row["operator_kind"] for row in rows]
    assert "ranking_top" in kinds
    assert "versus" in kinds
    assert "temporal_after" in kinds
    assert "numeric_less_than" in kinds
    assert {row["operator_class"] for row in rows} >= {"ranking", "versus", "temporal", "numeric"}
    assert all(isinstance(row["span"], list) and len(row["span"]) == 2 for row in rows)
    assert rows[0]["matched_text"] == "top 5 funds"


def test_query_comparison_operator_detector_case_insensitive_greater_and_bottom():
    rows = detect_query_comparison_operators("BOTTOM products with score GREATER THAN 90")

    assert [row["operator_kind"] for row in rows] == ["ranking_bottom", "numeric_greater_than"]
    assert rows[1]["confidence"] == 0.94
