from graph.store import summarize_unit_markdown_subscripts


def test_subscript_summary_counts_spans_and_ignores_strikethrough_and_fences():
    summary = summarize_unit_markdown_subscripts(
        [
            {"id": "b", "content": "~~strike~~\n~h2o~\n```\n~skip~\n```"},
            {"id": "a", "content": "~ion~ and ~ion~"},
        ],
        sample_limit=2,
    )
    assert summary["total_units"] == 2
    assert summary["units_with_subscript"] == 2
    assert summary["subscript_count"] == 3
    assert summary["most_common_text"] == "ion"
    assert summary["samples"] == [
        {"unit_id": "a", "line_number": 1, "text": "ion"},
        {"unit_id": "a", "line_number": 1, "text": "ion"},
    ]
