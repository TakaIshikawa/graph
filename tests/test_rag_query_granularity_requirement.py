from __future__ import annotations

from graph.rag.query_granularity_requirement import detect_query_granularity_requirement


def test_classifies_summary_detailed_stepwise_itemized_and_raw_data():
    assert detect_query_granularity_requirement("Briefly summarize it")["granularity"] == "summary"
    assert detect_query_granularity_requirement("Explain in detail")["granularity"] == "detailed"
    assert detect_query_granularity_requirement("Give me step by step instructions")["granularity"] == "stepwise"
    assert detect_query_granularity_requirement("Return line items")["granularity"] == "itemized"
    assert detect_query_granularity_requirement("Show raw records")["granularity"] == "raw_data"


def test_strongest_granularity_wins_with_evidence():
    report = detect_query_granularity_requirement("Briefly list line items and raw data rows")

    assert report["granularity"] == "raw_data"
    assert [row["granularity"] for row in report["matched_cues"]] == ["summary", "itemized", "raw_data"]


def test_blank_and_neutral_queries_are_unknown():
    assert detect_query_granularity_requirement("")["granularity"] == "unknown"
    assert detect_query_granularity_requirement("What happened?") == {
        "granularity": "unknown",
        "matched_cues": [],
        "has_granularity_requirement": False,
    }
