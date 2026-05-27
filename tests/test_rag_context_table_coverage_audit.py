from __future__ import annotations

from graph.rag.context_table_coverage import analyze_context_table_coverage


def test_detects_markdown_tables_in_context_content():
    report = analyze_context_table_coverage(
        [
            {
                "id": "md",
                "content": "| region | revenue |\n| --- | ---: |\n| US | 10 |\n| EU | 8 |",
            },
            {"id": "prose", "content": "Revenue improved in most regions."},
        ]
    )

    assert report["total_items"] == 2
    assert report["table_item_count"] == 1
    assert report["table_ratio"] == 0.5
    assert report["table_items"] == [{"item_id": "md", "index": 0, "table_type": "markdown"}]
    assert report["recommendation"] is None


def test_detects_csv_like_repeated_delimiter_rows():
    report = analyze_context_table_coverage(
        [{"source_id": "csv", "text": "service,p50,p95\napi,10,20\nweb,8,15"}]
    )

    assert report["table_item_count"] == 1
    assert report["table_ratio"] == 1.0
    assert report["table_items"] == [{"item_id": "csv", "index": 0, "table_type": "delimited"}]


def test_non_table_text_has_zero_table_coverage():
    report = analyze_context_table_coverage(
        [
            {"id": "a", "content": "The launch happened after planning, review, and approval."},
            {"id": "b", "snippet": "This paragraph has punctuation, but no repeated tabular rows."},
        ]
    )

    assert report == {
        "total_items": 2,
        "table_item_count": 0,
        "table_ratio": 0.0,
        "table_items": [],
        "recommendation": None,
    }


def test_recommends_table_evidence_for_numeric_table_oriented_queries_without_tables():
    report = analyze_context_table_coverage(
        [{"id": "summary", "content": "The API was faster after the cache rollout."}],
        query="Compare p95 latency by service in a table",
    )

    assert report["table_item_count"] == 0
    assert report["recommendation"] == "Add table-structured context for numeric or comparative queries before answering."
